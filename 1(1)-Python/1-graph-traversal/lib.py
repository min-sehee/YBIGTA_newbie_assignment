from __future__ import annotations
import copy
from collections import deque
from collections import defaultdict
from typing import DefaultDict, List


"""
TODO:
- __init__ 구현하기
- add_edge 구현하기
- dfs 구현하기 (재귀 또는 스택 방식 선택)
- bfs 구현하기
"""


class Graph:
    def __init__(self, n: int) -> None:
        """
        그래프 초기화

        Args:
            - n (int): 정점의 개수 (1번부터 n번까지)

        Attributes:
            - n (int): 정점의 개수
            - dict (DefaultDict[int, list[int]]): 각 정점과 연결된 인접 정점을 저장하는 딕셔너리
        """
        self.n = n
        self.dict: DefaultDict[int, list[int]] = defaultdict(list)

    
    def add_edge(self, u: int, v: int) -> None:
        """
        양방향 간선 추가

        Args:
            - u (int): 간선의 한쪽 끝 정점
            - v (int): 간선의 다른 한쪽 끝 정점
        """
        self.dict[u].append(v)
        self.dict[v].append(u)
    
    def dfs(self, start: int) -> list[int]:
        """
        깊이 우선 탐색 (DFS)
        
        구현 방법 선택:
        1. 재귀 방식: 함수 내부에서 재귀 함수 정의하여 구현
        2. 스택 방식: 명시적 스택을 사용하여 반복문으로 구현
        
        Args:
            - start (int): DFS를 시작할 정점 번호

        Returns:
            - result (list[int]): DFS 방문 순서를 저장하는 리스트
        
        """

        for key in self.dict:
            self.dict[key].sort()

        result = []

        def find(start: int) -> None:
            result.append(start)
            for i in self.dict[start]:
                if i not in result:
                    find(i)
                
        find(start)
        return result
    
    def bfs(self, start: int) -> list[int]:
        """
        너비 우선 탐색 (BFS)
        큐를 사용하여 구현

        Args:
            - start (int): BFS를 시작할 정점 번호

        Returns:
            - result (list[int]): BFS 방문 순서를 저장하는 리스트
        
        """

        for key in self.dict:
            self.dict[key].sort()

        result = []
        q: deque[int] = deque()

        result.append(start)
        q.append(start)
        
        while q:
            i = q.popleft()
            for i in self.dict[i]:
                if i not in result:
                    result.append(i)
                    q.append(i)

        return result
    
    def search_and_print(self, start: int) -> None:
        """
        DFS와 BFS 결과를 출력
        """
        dfs_result = self.dfs(start)
        bfs_result = self.bfs(start)
        
        print(' '.join(map(str, dfs_result)))
        print(' '.join(map(str, bfs_result)))
