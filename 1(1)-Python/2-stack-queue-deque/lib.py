from __future__ import annotations
from collections import deque


"""
TODO:
- rotate_and_remove 구현하기 
"""


def create_circular_queue(n: int) -> deque[int]:
    """1부터 n까지의 숫자로 deque를 생성합니다."""
    return deque(range(1, n + 1))

def rotate_and_remove(queue: deque[int], k: int) -> int:
    """
    큐에서 k번째 원소를 제거하고 반환합니다.

    Args:
        - queue (deque[int]): 현재 상태의 카드 큐
        - k (int): 제거할 카드의 위치

    Returns:
        - value (int): 제거된 카드의 번호
    """
    queue.rotate(-(k-1))    
    value = queue.popleft()
    
    return value