"""
Train Qwen2-VL-2B LoRA on VN plate crops (550 images).
Full model on GPU with gradient checkpointing — works on 4.3 GB VRAM.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent))

import gc, random, time, logging
from dataclasses import dataclass, field

import torch
import csv
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import LoraConfig, get_peft_model

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = _Path(__file__).parent.parent

SYSTEM_PROMPT = (
    "Ban la he thong nhan dien bien so xe Viet Nam. "
    "Doc va tra loi chi bien so xe, khong giai thich. "
    "Dinh dang: [ma tinh][chu cai loai][so]. Vi du: 30G112345"
)
USER_PROMPT = "Doc bien so xe trong anh nay:"


@dataclass
class TrainConfig:
    base_model: str = "Qwen/Qwen2-VL-2B-Instruct"
    output_dir: _Path = field(default_factory=lambda: PROJECT_ROOT / "experiments" / "qwen2vl_crops_lora")
    manifest: _Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "crops" / "manifest.csv")
    crops_dir: _Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "crops")
    epochs: int = 3
    per_device_batch: int = 1
    gradient_accumulation: int = 8
    max_seq_length: int = 512
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    seed: int = 42


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(manifest_path: _Path, crops_dir: _Path):
    rows = list(csv.DictReader(manifest_path.read_text(encoding="utf-8").strip().splitlines()))
    data = []
    for row in rows:
        img_path = crops_dir / f"{row['image_id']}.jpg"
        if img_path.exists():
            data.append({"image": str(img_path.resolve()), "plate": row["text_gt"]})
    logger.info("Loaded %d samples", len(data))
    return data


class VLMTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0

    def build_model(self):
        cfg = self.cfg
        logger.info("Loading base model (FP16, GPU, gradient checkpointing)...")
        t0 = time.time()
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            cfg.base_model,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
        )
        logger.info("Model loaded in %.0fs. VRAM: %.2f GB",
                     time.time() - t0, torch.cuda.memory_allocated() / 1e9)

        self.model.base_model._set_gradient_checkpointing(True)
        logger.info("Gradient checkpointing enabled")

        for name, param in self.model.named_parameters():
            if "visual" in name:
                param.requires_grad = False

        logger.info("Vision encoder frozen")

        lora_cfg = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.model.print_trainable_parameters()
        logger.info("LoRA applied. VRAM: %.2f GB", torch.cuda.memory_allocated() / 1e9)

        self.processor = Qwen2VLProcessor.from_pretrained(cfg.base_model)

    def build_optimizer(self, n_train: int):
        cfg = self.cfg
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        total_steps = (n_train // cfg.per_device_batch) * cfg.epochs // cfg.gradient_accumulation
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        from transformers import get_cosine_schedule_with_warmup
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        logger.info("Optimizer: lr=%.e, steps=%d (warmup=%d)",
                    cfg.learning_rate, total_steps, warmup_steps)

    def forward_batch(self, batch_data):
        cfg = self.cfg
        images = [Image.open(item["image"]).convert("RGB") for item in batch_data]
        plates = [item["plate"] for item in batch_data]

        texts = []
        for plate in plates:
            conv = [
                {"role": "user", "content": [
                    {"type": "image", "image": "placeholder"},
                    {"type": "text", "text": USER_PROMPT},
                ]},
                {"role": "assistant", "content": plate},
            ]
            text = self.processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

        inputs = self.processor(
            text=texts, images=images,
            return_tensors="pt", padding=True,
            truncation=True, max_length=cfg.max_seq_length,
        )
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        outputs = self.model(**inputs, labels=inputs["input_ids"])
        return outputs.loss

    def train_epoch(self, train_data: list, epoch: int):
        cfg = self.cfg
        random.shuffle(train_data)
        epoch_loss = 0.0
        n_processed = 0
        t0 = time.time()

        pbar = tqdm(
            range(0, len(train_data), cfg.per_device_batch),
            desc=f"Epoch {epoch+1}/{cfg.epochs}",
            ncols=80,
        )

        self.optimizer.zero_grad()

        for i in pbar:
            batch = train_data[i:i + cfg.per_device_batch]

            try:
                loss = self.forward_batch(batch)
                scaled = loss / cfg.gradient_accumulation
                scaled.backward()
            except Exception as e:
                logger.warning("Batch error: %s", e)
                gc.collect()
                torch.cuda.empty_cache()
                continue

            epoch_loss += loss.item() * len(batch)
            n_processed += len(batch)

            step = i // cfg.per_device_batch + 1
            if step % cfg.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                avg_loss = epoch_loss / max(n_processed, 1)
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                )

            del loss, scaled
            gc.collect()
            torch.cuda.empty_cache()

        # Flush remaining gradients
        remaining = n_processed % cfg.gradient_accumulation
        if remaining != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1

        elapsed = time.time() - t0
        return epoch_loss / max(n_processed, 1), elapsed

    def run(self):
        cfg = self.cfg
        set_seed(cfg.seed)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Training Qwen2-VL LoRA on VN plate crops")
        logger.info("=" * 60)
        logger.info("Config: epochs=%d, batch=%d, grad_accum=%d, lr=%.e",
                    cfg.epochs, cfg.per_device_batch, cfg.gradient_accumulation, cfg.learning_rate)
        logger.info("Output: %s", cfg.output_dir)

        train_data = load_data(cfg.manifest, cfg.crops_dir)
        self.build_model()
        self.build_optimizer(len(train_data))

        epoch_losses = []
        for epoch in range(cfg.epochs):
            avg_loss, elapsed = self.train_epoch(train_data, epoch)
            epoch_losses.append(round(avg_loss, 6))

            logger.info(
                "Epoch %d/%d | Loss=%.4f | Time=%.0fs | VRAM=%.2f GB",
                epoch + 1, cfg.epochs, avg_loss, elapsed,
                torch.cuda.memory_allocated() / 1e9,
            )

            ckpt_dir = cfg.output_dir / f"epoch-{epoch+1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(ckpt_dir)
            self.processor.save_pretrained(ckpt_dir)
            logger.info("Checkpoint saved: %s", ckpt_dir)

        # Save final
        logger.info("Saving final model...")
        self.model.save_pretrained(cfg.output_dir)
        self.processor.save_pretrained(cfg.output_dir)

        import json
        info = {
            "epochs": cfg.epochs,
            "per_device_batch": cfg.per_device_batch,
            "gradient_accumulation": cfg.gradient_accumulation,
            "learning_rate": cfg.learning_rate,
            "lora_r": 16, "lora_alpha": 32,
            "n_samples": len(train_data),
            "loss_per_epoch": epoch_losses,
            "total_steps": self.global_step,
        }
        (cfg.output_dir / "training_info.json").write_text(json.dumps(info, indent=2))

        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info("Loss per epoch: %s", [round(l, 4) for l in epoch_losses])
        logger.info("Output: %s", cfg.output_dir)
        logger.info("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "experiments" / "qwen2vl_crops_lora"))
    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.epochs = args.epochs
    cfg.per_device_batch = args.batch
    cfg.gradient_accumulation = args.grad_accum
    cfg.learning_rate = args.lr
    cfg.output_dir = _Path(args.output_dir)

    VLMTrainer(cfg).run()


if __name__ == "__main__":
    main()
