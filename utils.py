from torch.cuda import get_device_properties, max_memory_reserved
from transformers import BitsAndBytesConfig
import argparse

def byte_to_mega(byte):
    return round(byte / 1024 / 1024 / 1024, 3)

def get_current_gpu_memory(step_name="", device_idx=0):
    """
    현재 GPU 메모리 상태를 출력하는 함수.

    Args:
        step_name (str): 현재 시점을 나타내는 이름. 예: 'Training Step 1'.
    """
    gpu_stats = get_device_properties(device_idx)  # 0번째 GPU 속성 가져오기
    reserved_memory = byte_to_mega(max_memory_reserved())  # 예약된 GPU 메모리 (GB)
    total_memory = byte_to_mega(gpu_stats.total_memory)  # GPU의 총 메모리 (GB)

    # 현재 시점과 함께 메모리 상태 출력
    print(f"[{step_name}] {reserved_memory} GB of memory reserved.\n GPU = {gpu_stats.name}. Max memory = {total_memory} GB.")
    
    
def get_bnb_config(bit=8):
    if bit == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        print(f"You put {bit} bit in argument.\nWhatever the number you put in, if it is not 8 then 4bit config would be returned.")
        return BitsAndBytesConfig(load_in_4bit=True)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')