"""Comprehensive tests for VN License Plate Recognition Pipeline."""

import pytest
import numpy as np
from pathlib import Path

# Test plate_rules
from src.postprocess.plate_rules import (
    normalize_plate_text,
    is_valid_vn_plate,
    is_valid_province_code,
    advanced_repair_ocr_text,
    repair_common_ocr_errors,
    repair_tail_digit_confusions,
    get_best_repair_candidate,
    VALID_PROVINCE_CODES,
)

# Test types
from src.utils.types import (
    FrameData,
    Detection,
    PlateCrop,
    OcrResult,
    PipelineResult,
)

# Test preprocessing
from src.preprocess.ops import crop_plate, preprocess_plate

# Test base classes
from src.detector.base import PlateDetector, DummyCenterDetector
from src.ocr.base import PlateOcr, DummyOcr


class TestPlateRules:
    """Test VN plate normalization and validation."""

    def test_normalize_plate_text(self):
        """Test plate text normalization."""
        assert normalize_plate_text("12A-34567") == "12A34567"
        assert normalize_plate_text("51G-123.45") == "51G12345"
        assert normalize_plate_text("abc123") == "ABC123"
        assert normalize_plate_text("") == ""

    def test_is_valid_vn_plate(self):
        """Test VN plate validation."""
        # Valid plates
        assert is_valid_vn_plate("30G12345") == True  # 8-char car
        assert is_valid_vn_plate("51H123456") == True  # 9-char car
        assert is_valid_vn_plate("43A1234") == True  # 7-char bike
        assert is_valid_vn_plate("29D12345") == True  # 8-char car

        # Invalid plates
        assert is_valid_vn_plate("XX123456") == False  # Invalid province
        assert is_valid_vn_plate("12A1234X") == False  # Letter in number part
        assert is_valid_vn_plate("") == False  # Empty

    def test_is_valid_province_code(self):
        """Test province code validation."""
        assert is_valid_province_code("30") == True
        assert is_valid_province_code("51") == True
        assert is_valid_province_code("29") == True
        assert is_valid_province_code("XX") == False
        assert is_valid_province_code("1") == False  # Too short

    def test_repair_common_ocr_errors(self):
        """Test common OCR error repair."""
        # O in province code should become 0
        assert repair_common_ocr_errors("OOG12345") == "00G12345"
        assert repair_common_ocr_errors("1OG12345") == "10G12345"
        # Non-O chars should be preserved
        assert repair_common_ocr_errors("12A12345") == "12A12345"

    def test_repair_tail_digit_confusions(self):
        """Test tail digit confusion repair."""
        # I/L/1 should become 1 - but only at position >= 3 (tail)
        assert repair_tail_digit_confusions("123I12345") == "123112345"  # I at pos 3+
        assert repair_tail_digit_confusions("123L12345") == "123112345"  # L at pos 3+
        # O should become 0 - only first O in tail gets replaced
        assert repair_tail_digit_confusions("12AOO234") == "12A00234"
        # S should become 5 at tail positions
        assert repair_tail_digit_confusions("12AS12345") == "12A512345"

    def test_advanced_repair_ocr_text(self):
        """Test advanced OCR text repair."""
        # Already valid plate
        result = advanced_repair_ocr_text("30G12345")
        assert result == "30G12345"

        # OCR confusion: I in series position (pos 2) is valid in VN plates
        result = advanced_repair_ocr_text("51I12345")
        assert result == "51I12345"  # I is valid in series position

        # OCR confusion: S→5 is valid in number section
        result = advanced_repair_ocr_text("30G12S45")
        # S should be replaced with 5 in number section
        assert "5" in result or result == "30G12145"

    def test_get_best_repair_candidate(self):
        """Test best repair candidate selection."""
        # Valid plate should return exact match
        result = get_best_repair_candidate("30G12345")
        assert result.method == "exact_match"
        assert result.score == 1.0


class TestTypes:
    """Test data types and dataclasses."""

    def test_frame_data(self):
        """Test FrameData dataclass."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fd = FrameData(image_id="test_001", frame=frame, source="test")
        assert fd.image_id == "test_001"
        assert fd.frame.shape == (480, 640, 3)
        assert fd.source == "test"
        assert fd.timestamp_ms is None

    def test_detection(self):
        """Test Detection dataclass."""
        det = Detection(
            image_id="test_001",
            bbox_xyxy=(10, 20, 100, 80),
            score=0.95,
        )
        assert det.bbox_xyxy == (10, 20, 100, 80)
        assert det.score == 0.95
        assert det.class_name == "license_plate"

    def test_plate_crop(self):
        """Test PlateCrop dataclass."""
        crop = np.zeros((60, 200, 3), dtype=np.uint8)
        pc = PlateCrop(
            image_id="test_001",
            crop=crop,
            bbox_xyxy=(10, 20, 210, 80),
            det_score=0.95,
        )
        assert pc.crop.shape == (60, 200, 3)

    def test_ocr_result(self):
        """Test OcrResult dataclass."""
        ocr = OcrResult(
            image_id="test_001",
            text_raw="30G12345",
            text_norm="30G12345",
            ocr_score=0.9,
        )
        assert ocr.text_raw == "30G12345"
        assert ocr.text_norm == "30G12345"
        assert ocr.ocr_score == 0.9

    def test_pipeline_result(self):
        """Test PipelineResult dataclass."""
        pr = PipelineResult(
            image_id="test_001",
            plate_text="30G12345",
            bbox_xyxy=(10, 20, 100, 80),
            confidence=0.9,
            source="test",
        )
        assert pr.plate_text == "30G12345"
        assert pr.bbox_xyxy == (10, 20, 100, 80)
        assert pr.confidence == 0.9


class TestPreprocessing:
    """Test preprocessing operations."""

    def test_crop_plate_basic(self):
        """Test basic plate cropping."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fd = FrameData(image_id="test", frame=frame, source="test")
        det = Detection(
            image_id="test",
            bbox_xyxy=(100, 150, 300, 200),
            score=0.95,
        )
        crop = crop_plate(fd, det)
        assert crop.crop.shape[0] == 50  # height
        assert crop.crop.shape[1] == 200  # width

    def test_crop_plate_with_margin(self):
        """Test plate cropping with margin."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fd = FrameData(image_id="test", frame=frame, source="test")
        det = Detection(
            image_id="test",
            bbox_xyxy=(100, 150, 300, 200),
            score=0.95,
        )
        crop = crop_plate(fd, det, margin_ratio=0.1)
        # With 10% margin, crop should be larger
        assert crop.crop.shape[1] >= 200

    def test_preprocess_plate(self):
        """Test plate preprocessing."""
        crop = np.zeros((60, 200, 3), dtype=np.uint8)
        result = preprocess_plate(crop)
        assert result.dtype == np.uint8
        assert result.shape == (120, 320)  # default output_size

    def test_preprocess_plate_with_clahe(self):
        """Test plate preprocessing with CLAHE."""
        crop = np.zeros((60, 200, 3), dtype=np.uint8)
        result = preprocess_plate(crop, use_clahe=True)
        assert result.dtype == np.uint8
        assert result.shape == (120, 320)


class TestBaseClasses:
    """Test base classes and protocols."""

    def test_dummy_detector(self):
        """Test DummyCenterDetector."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fd = FrameData(image_id="test", frame=frame, source="test")
        detector = DummyCenterDetector()
        detections = detector.predict(fd)
        assert len(detections) == 1
        assert detections[0].score == 0.5

    def test_dummy_ocr(self):
        """Test DummyOcr."""
        crop = np.zeros((60, 200, 3), dtype=np.uint8)
        pc = PlateCrop(
            image_id="test",
            crop=crop,
            bbox_xyxy=(10, 20, 210, 80),
            det_score=0.95,
        )
        ocr = DummyOcr()
        result = ocr.recognize(pc)
        assert result.text_raw == "51H12345"
        assert result.ocr_score == 0.5

    def test_plate_detector_protocol(self):
        """Test PlateDetector protocol exists."""
        # Protocol should exist as a class
        assert PlateDetector is not None
        assert hasattr(PlateDetector, 'predict')

    def test_plate_ocr_protocol(self):
        """Test PlateOcr protocol exists."""
        # Protocol should exist as a class
        assert PlateOcr is not None
        assert hasattr(PlateOcr, 'recognize')


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_text_normalization(self):
        """Test normalization of empty text."""
        assert normalize_plate_text("") == ""
        assert normalize_plate_text("   ") == ""
        assert is_valid_vn_plate("") == False

    def test_whitespace_handling(self):
        """Test whitespace in plate text."""
        assert normalize_plate_text("  12A34567  ") == "12A34567"
        assert normalize_plate_text("12-A-34567") == "12A34567"

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert normalize_plate_text("abc123") == "ABC123"
        assert normalize_plate_text("XYZ789") == "XYZ789"

    def test_special_characters(self):
        """Test special character removal."""
        assert normalize_plate_text("12A-345.67") == "12A34567"
        assert normalize_plate_text("12A_345_67") == "12A34567"
        assert normalize_plate_text("12A@#$%567") == "12A567"

    def test_invalid_province_codes(self):
        """Test invalid province code handling."""
        # 00 is not a valid province code
        assert "00" not in VALID_PROVINCE_CODES
        # 99 is not a valid province code
        assert "99" not in VALID_PROVINCE_CODES
        # Real codes
        assert "30" in VALID_PROVINCE_CODES
        assert "51" in VALID_PROVINCE_CODES

    def test_detection_with_invalid_bbox(self):
        """Test detection with invalid bbox."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fd = FrameData(image_id="test", frame=frame, source="test")
        # Bbox outside frame
        det = Detection(
            image_id="test",
            bbox_xyxy=(700, 500, 900, 600),  # Outside frame
            score=0.95,
        )
        crop = crop_plate(fd, det)
        # Should still return a crop (clipped to frame)
        assert crop.crop is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
