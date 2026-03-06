import torch
import time

def check_torch_cuda():
    print(f"--- PyTorch CUDA 檢測 ---")
    print(f"PyTorch 版本: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 是否可用: {cuda_available}")
    
    if cuda_available:
        print(f"當前設備: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")
        
        # 進行一次矩陣運算測試性能
        size = 1000
        a = torch.randn(size, size).to("cuda")
        b = torch.randn(size, size).to("cuda")
        
        # 預熱
        _ = torch.matmul(a, b)
        
        start = time.time()
        for _ in range(100):
            c = torch.matmul(a, b)
        torch.cuda.synchronize() # 等待 GPU 算完
        end = time.time()
        
        print(f"🚀 100 次矩陣相乘耗時: {(end - start):.4f}s")
        print(f"如果耗時 < 0.1s，說明 CUDA 運行完美！")
    else:
        print("❌ 錯誤：PyTorch 目前只能使用 CPU，這就是 FPS 低的原因。")

if __name__ == "__main__":
    check_torch_cuda()