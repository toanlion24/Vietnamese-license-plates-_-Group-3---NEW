// @ts-nocheck
import { BarChart, Card, CardBody, CardHeader, Code, Divider, Grid, H1, H2, H3, Link, Pill, Row, Stack, Stat, Table, Text, useHostTheme } from "cursor/canvas";

const metricRows = [
  ["train_box_loss", "0.72458", "0.40818", "Giảm 43.7%", "Hộp dự đoán khớp nhãn tốt hơn"],
  ["train_cls_loss", "1.28634", "0.20274", "Giảm 84.2%", "Mô hình tự tin hơn với lớp license_plate"],
  ["precision", "0.94090", "0.98852", "Tăng 5.1%", "Ít báo nhầm vùng không phải biển số"],
  ["recall", "0.95620", "0.98515", "Tăng 3.0%", "Ít bỏ sót biển số thật"],
  ["mAP50", "0.97145", "0.99382", "Tăng 2.3%", "Định vị tốt ở ngưỡng IoU 0.50"],
  ["mAP50-95", "0.75854", "0.90098", "Tăng 18.8%", "Hộp dự đoán chính xác hơn ở nhiều ngưỡng IoU"],
];

const flowSteps = [
  { title: "Dữ liệu", body: "data/images/raw, data/labels/raw, train/val/test split" },
  { title: "data.yaml", body: "Khai báo đường dẫn và lớp 0: license_plate cho YOLO" },
  { title: "Train YOLOv8n", body: "50 epoch, ảnh 640, batch 16, pretrained yolov8n.pt" },
  { title: "Checkpoint", body: "Lấy best.pt và copy thành weights/yolov8_license_plate.pt" },
  { title: "Inference", body: "Chạy 20 ảnh demo bằng detector YOLO + OCR dummy" },
  { title: "Đánh giá lỗi", body: "So bbox dự đoán với ground truth bằng IoU và ảnh bbox" },
];

const formulaRows = [
  ["IoU", "IoU = Area(A ∩ B) / Area(A ∪ B)", "Đo độ trùng nhau giữa bbox dự đoán và bbox thật. Gần 1 là khoanh rất sát."],
  ["Precision", "TP / (TP + FP)", "Trong các bbox model dự đoán là biển số, có bao nhiêu bbox thật sự đúng."],
  ["Recall", "TP / (TP + FN)", "Trong tất cả biển số thật, model tìm được bao nhiêu biển số."],
  ["F1-score", "2PR / (P + R)", "Gộp Precision và Recall thành một điểm cân bằng."],
  ["AP", "Diện tích dưới đường Precision-Recall", "Đánh giá một lớp object detection tại một ngưỡng IoU."],
  ["mAP50", "mean(AP tại IoU = 0.50)", "Chỉ cần bbox trùng từ 50% trở lên, tương đối dễ đạt."],
  ["mAP50-95", "mean(AP tại IoU 0.50, 0.55, ..., 0.95)", "Khắt khe hơn vì bbox phải chính xác ở nhiều mức độ."],
];

const algorithmRows = [
  ["Backbone", "Trích xuất đặc trưng từ ảnh", "Từ pixel thô, model học ra cạnh, góc, hình dạng biển số."],
  ["Neck", "Ghép đặc trưng nhiều kích thước", "Giúp phát hiện cả biển số nhỏ và biển số lớn."],
  ["Head", "Dự đoán bbox, class, score", "Với mỗi vùng nghi ngờ, YOLO trả về tọa độ và độ tin cậy."],
  ["Loss", "So sánh dự đoán với nhãn thật", "Train là quá trình làm loss nhỏ dần để bbox/class tốt hơn."],
  ["NMS", "Giữ bbox tốt nhất, bỏ bbox trùng", "Nếu nhiều khung cùng khoanh một biển số, chỉ giữ khung có score cao."],
];

function FlowMap() {
  const theme = useHostTheme();
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 10 }}>
      {flowSteps.map((step, index) => (
        <div
          key={step.title}
          style={{
            border: `1px solid ${theme.stroke.secondary}`,
            background: index === 2 ? theme.fill.secondary : theme.fill.tertiary,
            padding: 12,
            borderRadius: 8,
            minHeight: 108,
          }}
        >
          <Text size="small" tone="secondary">Bước {index + 1}</Text>
          <H3>{step.title}</H3>
          <Text size="small">{step.body}</Text>
        </div>
      ))}
    </div>
  );
}

function MiniLegend() {
  const theme = useHostTheme();
  return (
    <Row gap={12} wrap>
      <Row gap={6} align="center">
        <span style={{ width: 18, height: 10, background: theme.palette.green9, display: "inline-block" }} />
        <Text size="small">Xanh lá: bbox ground truth</Text>
      </Row>
      <Row gap={6} align="center">
        <span style={{ width: 18, height: 10, background: theme.palette.red9, display: "inline-block" }} />
        <Text size="small">Đỏ: bbox mô hình dự đoán</Text>
      </Row>
    </Row>
  );
}

export default function Buoi3YoloBaselineExplainer() {
  return (
    <Stack gap={18}>
      <H1>Buổi 3: Hiểu Notebook Detection Baseline YOLO</H1>
      <Text>
        Notebook này xây dựng baseline phát hiện vị trí biển số bằng YOLOv8n. Mục tiêu không phải đọc chữ trên biển số, mà là cắt đúng vùng biển số để chuẩn bị cho OCR ở các buổi sau.
      </Text>

      <Grid columns={4} gap={12}>
        <Stat value="3204" label="ảnh train" />
        <Stat value="915" label="ảnh validation" />
        <Stat value="50" label="epoch trong kết quả metrics" />
        <Stat value="0.901" label="mAP50-95 cuối" tone="success" />
      </Grid>

      <Divider />

      <H2>Sơ Đồ Tư Duy Quy Trình</H2>
      <FlowMap />

      <Grid columns={2} gap={16}>
        <Stack gap={10}>
          <H2>Metrics Chính</H2>
          <BarChart
            categories={["Precision", "Recall", "mAP50", "mAP50-95"]}
            series={[
              { name: "Epoch 1", data: [94.09, 95.62, 97.15, 75.85], tone: "neutral" },
              { name: "Epoch 50", data: [98.85, 98.52, 99.38, 90.10], tone: "success" },
            ]}
            height={260}
            valueSuffix="%"
          />
          <Text size="small" tone="secondary">
            mAP50-95 tăng mạnh nhất vì đây là chỉ số khắt khe hơn mAP50: bbox phải khớp tốt qua nhiều ngưỡng IoU từ 0.50 đến 0.95.
          </Text>
        </Stack>

        <Card>
          <CardHeader>Đọc Hình Bbox Trong Notebook</CardHeader>
          <CardBody>
            <Stack gap={10}>
              <MiniLegend />
              <Text>
                Nếu khung đỏ gần trùng khung xanh và IoU lớn hơn 0.5, detection được xem là đạt. Trong 12 ảnh notebook in ra, IoU trung bình khoảng 0.933 và tất cả được phân loại là "detect ổn".
              </Text>
              <Pill tone="success">12/12 mẫu kiểm tra nhanh detect ổn</Pill>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <H2>Bảng Giải Thích Số Liệu</H2>
      <Table
        headers={["Chỉ số", "Epoch 1", "Epoch 50", "Thay đổi", "Ý nghĩa"]}
        rows={metricRows}
      />

      <H2>Thuật Toán Và Công Thức Toán Học</H2>
      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>YOLO Detect Biển Số Như Thế Nào?</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>
                YOLO là thuật toán object detection một giai đoạn: ảnh đi qua mạng neural một lần, sau đó model dự đoán trực tiếp bbox, lớp vật thể và điểm tin cậy.
              </Text>
              <Table
                headers={["Thành phần", "Vai trò", "Hiểu đơn giản"]}
                rows={algorithmRows}
              />
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>Công Thức Cần Nhớ</CardHeader>
          <CardBody>
            <Table
              headers={["Tên", "Công thức", "Ý nghĩa"]}
              rows={formulaRows}
            />
          </CardBody>
        </Card>
      </Grid>

      <Card>
        <CardHeader>Ví Dụ Cực Dễ Hiểu</CardHeader>
        <CardBody>
          <Stack gap={8}>
            <Text>
              Giả sử có 100 ảnh có biển số thật. Model khoanh 98 vùng là biển số, trong đó 97 vùng đúng và 1 vùng nhầm nền.
            </Text>
            <Text>
              Khi đó: <Code>TP=97</Code>, <Code>FP=1</Code>, <Code>FN=3</Code>. Precision = 97/(97+1) = 0.990; Recall = 97/(97+3) = 0.970.
            </Text>
            <Text>
              Nếu bbox đỏ gần trùng bbox xanh thì IoU cao. Nếu IoU đạt ngưỡng, dự đoán được tính là đúng; nếu không, nó bị xem là detect lệch hoặc sai vùng.
            </Text>
          </Stack>
        </CardBody>
      </Card>

      <Grid columns={3} gap={12}>
        <Stat value="20" label="ảnh inference demo" />
        <Stat value="99.6 ms" label="latency trung bình/ảnh" />
        <Stat value="85.9 ms" label="latency median/ảnh" />
      </Grid>

      <Card>
        <CardHeader>Nhận Xét Đánh Giá</CardHeader>
        <CardBody>
          <Stack gap={8}>
            <Text>Baseline đang tốt cho bài toán định vị biển số: precision, recall và mAP đều cao, đặc biệt mAP50 đạt 0.99382 ở epoch 50.</Text>
            <Text>Điểm cần cẩn trọng: inference trong notebook dùng <Code>ocr-backend dummy</Code>, nên <Code>plate_text</Code> chỉ là giá trị giả để kiểm thử pipeline, chưa đánh giá nhận dạng ký tự thật.</Text>
            <Text>Train log trong cell hiện có đoạn bị ngắt bằng KeyboardInterrupt trên CPU, nhưng cell metrics phía sau đọc được 50 dòng kết quả từ một run đã hoàn tất trước đó.</Text>
          </Stack>
        </CardBody>
      </Card>

      <H2>Tài Liệu Học Tập</H2>
      <Stack gap={6}>
        <Link href="https://docs.ultralytics.com/tasks/detect/">Ultralytics YOLO Detection Documentation</Link>
        <Link href="https://docs.ultralytics.com/guides/yolo-performance-metrics/">Ultralytics: YOLO Performance Metrics</Link>
        <Link href="https://cocodataset.org/#detection-eval">COCO Detection Evaluation Metrics</Link>
      </Stack>
    </Stack>
  );
}
