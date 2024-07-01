import os
from mmcv import Config
from mmdet.datasets import build_dataset, build_dataloader
from mmcv.parallel import DataContainer as DC
from PIL import Image, ImageDraw, ImageFont
import numpy as np




def unwrap_data(data):
    if isinstance(data, DC):
        return data.data[0] if isinstance(data.data, list) else data.data
    return data

class ImageBatchProcessor:
    def __init__(self, class_names, output_dir='examples'):
        self.class_names = class_names
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def process_batch(self, images, annotations, labels):
        batch_size = images.shape[0]
        height, width = images.shape[2], images.shape[3]

        num_cols = min(batch_size, 4)  # 최대 4열로 설정
        num_rows = (batch_size - 1) // num_cols + 1  # 행 수 계산

        total_width = width * num_cols
        total_height = height * num_rows
        combined_image = Image.new('RGB', (total_width, total_height))

        font_size = min(width, height) // 20  # 이미지 크기에 따라 폰트 크기 조정

        for j in range(batch_size):
            img = images[j].cpu().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min()) * 255  # 정규화 해제 및 스케일링
            img = img.astype(np.uint8)
            pil_img = Image.fromarray(img)

            col_idx = j % num_cols
            row_idx = j // num_cols
            combined_image.paste(pil_img, (col_idx * width, row_idx * height))

            draw = ImageDraw.Draw(combined_image)
            for bbox, label in zip(annotations[j], labels[j]):
                bbox = bbox.cpu().numpy()
                label = label.cpu().numpy()
                x1, y1, x2, y2 = bbox
                draw.rectangle([((col_idx * width) + x1, (row_idx * height) + y1),
                                ((col_idx * width) + x2, (row_idx * height) + y2)],
                               outline='red', width=2)

                text = self.class_names[label]

                font = ImageFont.truetype("arial.ttf", font_size)

                text_left, text_top, text_right, text_bottom = draw.textbbox((0, 0), text, font=font)
                text_width = text_right - text_left
                text_height = text_bottom - text_top
                text_x = (col_idx * width) + x1 + (x2 - x1 - text_width) // 2
                text_y = (row_idx * height) + y1 - text_height - 1

                draw.text((text_x, text_y), text, fill='red', font=font)

        return combined_image

    def save_combined_image(self, combined_image, filename='combined_batch_image.jpg'):
        combined_image.save(os.path.join(self.output_dir, filename))

def main():
    # Config 파일을 로드합니다.
    cfg = Config.fromfile('config.py')

    # 데이터셋을 빌드합니다.
    datasets = [build_dataset(cfg.data.train)]

    # 데이터 로더를 빌드합니다.
    data_loaders = [
        build_dataloader(
            ds,
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            num_gpus=1,  # 사용할 GPU 수
            dist=False,  # 분산 학습 여부
            shuffle=True) for ds in datasets
    ]

    # 클래스 이름을 가져옵니다.
    class_names = datasets[0].CLASSES

    processor = ImageBatchProcessor(class_names)

    for i, data in enumerate(data_loaders[0]):
        if i == 0:  # 첫 번째 배치만 가져오기
            images = unwrap_data(data['img'])
            annotations = unwrap_data(data['gt_bboxes'])
            labels = unwrap_data(data['gt_labels'])

            combined_image = processor.process_batch(images, annotations, labels)
            processor.save_combined_image(combined_image)
            break

if __name__ == '__main__':
    main()

