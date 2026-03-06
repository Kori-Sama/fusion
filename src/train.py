import torch
from torch.utils.data import DataLoader
from src.data.nuscenes_dataset import NuscenesDataset
from src.models.centernet3d import get_model
from src.loss.losses import CenterFusionLoss

def train(data_root, version='v1.0-mini', epochs=50, batch_size=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载数据集
    dataset = NuscenesDataset(data_root, version=version, split='train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 2. 初始化模型
    model = get_model(num_classes=10).to(device)
    
    # 3. 初始化优化器与损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1.25e-4)
    criterion = CenterFusionLoss()
    
    # 4. 训练循环
    for epoch in range(epochs):
        model.train()
        for i, (imgs, targets) in enumerate(dataloader):
            imgs = imgs.to(device)
            # 将 targets 中的 Tensor 转到 device
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # 正向推理 (图像 + 雷达图)
            outputs = model(imgs, targets['radar_hm'])
            
            # 计算损失
            loss, loss_stats = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] "
                      f"Total Loss: {loss_stats['total']:.4f} HM Loss: {loss_stats['hm']:.4f}")

if __name__ == "__main__":
    # 示例运行 (请确保已下载 nuScenes 数据集)
    train(data_root='/path/to/nuscenes/data')
