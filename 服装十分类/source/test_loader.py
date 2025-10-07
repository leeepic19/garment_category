import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from source.main import *
import matplotlib.pyplot as plt

# Fashion-MNIST 类别标签
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

if __name__ == '__main__':
    # 获取一个批次的数据
    images, labels = next(iter(train_loader))
    print(f"Batch shape: images={images.shape}, labels={labels.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")

    # 可视化前16张图片
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle('Fashion-MNIST Sample Images', fontsize=16)

    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            # 显示图片 (去掉channel维度)
            ax.imshow(images[idx].squeeze(), cmap="gray")
            ax.set_title(f"{class_names[labels[idx]]}", fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(str(project_root / 'data_visualization.png'), dpi=100, bbox_inches='tight')
    print(f"\n✓ 可视化结果已保存到: {project_root / 'data_visualization.png'}")
    plt.show()