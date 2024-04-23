import os
from PIL import Image

def split_image(image_path, output_dir, block_size):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 打开原始图像
    image = Image.open(image_path)

    # 获取原始图像的宽度和高度
    width, height = image.size

    # 计算图像块的行数和列数
    rows = height // block_size
    cols = width // block_size

    # 获取原始文件名
    file_name = os.path.splitext(os.path.basename(image_path))[0]

    # 分割图像并保存
    count = 0
    for row in range(rows):
        for col in range(cols):
            # 计算当前图像块的左上角和右下角坐标
            left = col * block_size
            top = row * block_size
            right = left + block_size
            bottom = top + block_size

            # 裁剪图像块
            block = image.crop((left, top, right, bottom))

            # 生成新的文件名
            new_file_name = f"{file_name}_block_{count}-{row}-{col}.png"

            # 保存图像块
            block_path = os.path.join(output_dir, new_file_name)
            block.save(block_path)

            count += 1

    print(f"成功分割图像，共得到 {count} 个图像块。")

# 设置输入路径和输出目录
input_path = "G:\\OneDrive\\Desktop\\ECDataset2\\POLE-mut"
output_directory = "G:\\EC\\TEST\\POLE-mut"

# 设置图像块的大小
block_size = 224

# 获取输入路径下的所有图像文件
image_files = [f for f in os.listdir(input_path) if f.endswith(".png") or f.endswith(".jpg")]

# 遍历每个图像文件并分割图像
for image_file in image_files:
    image_path = os.path.join(input_path, image_file)
    split_image(image_path, output_directory, block_size)
