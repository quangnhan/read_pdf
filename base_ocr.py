from enum import Enum
from itertools import product
from pathlib import Path
from typing import NamedTuple

import cv2 as cv
import numpy as np
from numpy import ndarray
from pdf2image import convert_from_path
from pytesseract import image_to_string

OCR_DEBUG = False


class OcrLanguage(str, Enum):
    ENGLISH    = 'eng'
    VIETNAMESE = 'vie'


class Rectangle(NamedTuple):
    x     : int
    y     : int
    width : int
    height: int

    def __str__(self):
        return f'Rectangle({self.x}, {self.y}, {self.width}, {self.height})'


# def read_text(binary_img: ndarray, language: OcrLanguage = OcrLanguage.VIETNAMESE) -> str:
#     return image_to_string(binary_img, lang=language.value)


def get_red_channel(bgr_img: ndarray) -> ndarray:
    return bgr_img[:, :, 2]


def get_green_channel(bgr_img: ndarray) -> ndarray:
    return bgr_img[:, :, 1]


def get_blue_channel(bgr_img: ndarray) -> ndarray:
    return bgr_img[:, :, 0]


def get_roi(img: ndarray, rect: Rectangle) -> ndarray:
    x, y, w, h = rect
    return img[y:y + h, x:x + w]


def detect_horizontal_lines(binary_img: ndarray, min_length: int, min_width: int):
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (min_length, min_width))
    detected_lines = cv.morphologyEx(binary_img, cv.MORPH_OPEN, horizontal_kernel, iterations=2)
    lines = cv.findContours(detected_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # lines                  List of detected lines
    # lines[.]               Detected line, contains 4 points define the area of the lines (from top left, and counter-clockwise)
    # lines[.][0-3]          1 of 4 points define the line area (it only contains 1 value: [[x, y]])
    # lines[.][0-3][0]       1 of 4 points define the line area (it contains the value of the points [x, y])
    # lines[.][0-3][0][0-1]  [0]: x value, [1]: y value
    return sorted(lines[0], key=lambda x: x[0][0][1])


def detect_vertical_lines(binary_img: ndarray, min_length: int, min_width: int):
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (min_width, min_length))
    detected_lines = cv.morphologyEx(binary_img, cv.MORPH_OPEN, vertical_kernel, iterations=2)
    lines = cv.findContours(detected_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # lines                  List of detected lines
    # lines[.]               Detected line, contains 4 points define the area of the lines (from top left, and counter-clockwise)
    # lines[.][0-3]          1 of 4 points define the line area (it only contains 1 value: [[x, y]])
    # lines[.][0-3][0]       1 of 4 points define the line area (it contains the value of the points [x, y])
    # lines[.][0-3][0][0-1]  [0]: x value, [1]: y value
    return sorted(lines[0], key=lambda x: x[0][0][0])


def detect_table(gray_img: ndarray, rect: Rectangle):
    MIN_HORIZONTAL_LINE = 200
    MIN_VERTICAL_LINE   = 50
    LINE_WIDTH          = 3
    x, y, w, h = rect
    roi_img = gray_img[y:y + h, x:x + w]
    _, binary_img = cv.threshold(roi_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    inverted_binary_img = np.invert(binary_img)
    # cv.imshow('Binary', binary_img)
    # cv.imshow('Inverted Binary', inverted_binary_img)

    # Horizontal Lines
    horizontal_lines = detect_horizontal_lines(inverted_binary_img, MIN_HORIZONTAL_LINE, LINE_WIDTH)
    vertical_lines = detect_vertical_lines(inverted_binary_img, MIN_VERTICAL_LINE, LINE_WIDTH)
    total_rows    = len(horizontal_lines) - 1
    total_columns = len(vertical_lines)   - 1
    # Debugging
    result_img = cv.cvtColor(roi_img, cv.COLOR_GRAY2BGR)
    # for h,v in product(horizontal_lines, vertical_lines):
    #     cv.drawContours(result_img, [h], -1, (0, 0, 180), 3)
    #     cv.drawContours(result_img, [v], -1, (0, 0, 180), 3)
    for y, x in product(range(total_rows), range(total_columns)):
        y1 = horizontal_lines[y][1][0][1] + 4
        y2 = horizontal_lines[y + 1][0][0][1] - 4
        x1 = vertical_lines[x][2][0][0] + 4
        x2 = vertical_lines[x + 1][1][0][0] - 4
        result_img = cv.rectangle(result_img, (x1, y1), (x2, y2), (100, 250, 100), 3)
    cv.imshow('Lines Detected', result_img)
    if cv.waitKey(-1) == ord('q'):
        cv.destroyAllWindows()
        OCR_DEBUG = False


def read_table(gray_img: ndarray, rect: Rectangle, language: OcrLanguage = OcrLanguage.ENGLISH) -> list[list[str]]:
    global OCR_DEBUG
    if OCR_DEBUG:
        detect_table(gray_img, rect)
    MIN_HORIZONTAL_LINE = 200
    MIN_VERTICAL_LINE   = 50
    LINE_WIDTH          = 3
    x, y, w, h = rect
    roi_img = gray_img[y:y + h, x:x + w]
    _, binary_img = cv.threshold(roi_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    inverted_binary_img = np.invert(binary_img)
    # Horizontal & Vertical lines
    horizontal_lines = detect_horizontal_lines(inverted_binary_img, MIN_HORIZONTAL_LINE, LINE_WIDTH)
    vertical_lines = detect_vertical_lines(inverted_binary_img, MIN_VERTICAL_LINE, LINE_WIDTH)
    total_rows    = len(horizontal_lines) - 1
    total_columns = len(vertical_lines)   - 1
    # Detect texts between lines
    texts = [[''] * total_columns for _ in range(total_rows)]
    for y in range(total_rows):
        for x in range(total_columns):
            y1 = horizontal_lines[y    ][1][0][1] + 1
            y2 = horizontal_lines[y + 1][0][0][1]
            x1 = vertical_lines  [x    ][2][0][0] + 1
            x2 = vertical_lines  [x + 1][1][0][0]
            roi_img = binary_img[y1:y2, x1:x2]
            texts[y][x] = image_to_string(roi_img, lang=language.value).strip()
    return texts


def ocr_text_preprocess(text: str):
    return text.replace('\n\n', '\n').replace(',\n', ',').replace(' :', ':').strip()


def read_text(gray_img: ndarray, rect: Rectangle, language: OcrLanguage = OcrLanguage.ENGLISH) -> str:
    x, y, w, h = rect
    roi_img = gray_img[y: y + h, x: x + w]
    _, binary_img = cv.threshold(roi_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    global OCR_DEBUG
    if OCR_DEBUG:
        cv.imshow('Extracted', binary_img)
        if cv.waitKey(-1) == ord('q'):
            cv.destroyAllWindows()
            OCR_DEBUG = False
    info = image_to_string(binary_img, lang=language.value)
    return ocr_text_preprocess(info)


def find_line_contain(text_lines: list[str], sub_texts: list[str]):
    sub_texts = [sub_text.upper() for sub_text in sub_texts]
    for i, line in enumerate(text_lines):
        line = line.upper()
        if any(sub_text in line for sub_text in sub_texts):
            return i
    return -1


def read_pdf(pdf_file: str, dpi: int = 400):
    try:
        imgs = convert_from_path(pdf_file, dpi)
    except Exception as ex:
        print(ex)
        return []
    return [np.array(img, dtype=np.uint8) for img in imgs]


def convert_pdf_to_images(pdf_file: str, output_path: str):
    filename = Path(pdf_file).stem
    images = convert_from_path(pdf_file, 400)
    for i, image in enumerate(images):
        image.save(f'{output_path}//{filename}_page_{i}.jpg', 'JPEG')


def read_text_auto(bgr_img: ndarray, language: OcrLanguage = OcrLanguage.VIETNAMESE) -> str:
    gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
    _, binary_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    text = image_to_string(binary_img, lang=language.value)
    return ocr_text_preprocess(text)
