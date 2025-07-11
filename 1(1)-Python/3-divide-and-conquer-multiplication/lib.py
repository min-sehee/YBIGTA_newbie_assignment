from __future__ import annotations
import copy


"""
TODO:
- __setitem__ 구현하기
- __pow__ 구현하기 (__matmul__을 활용해봅시다)
- __repr__ 구현하기
"""


class Matrix:
    MOD = 1000

    def __init__(self, matrix: list[list[int]]) -> None:
        self.matrix = matrix

    @staticmethod
    def full(n: int, shape: tuple[int, int]) -> Matrix:
        return Matrix([[n] * shape[1] for _ in range(shape[0])])

    @staticmethod
    def zeros(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(0, shape)

    @staticmethod
    def ones(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(1, shape)

    @staticmethod
    def eye(n: int) -> Matrix:
        matrix = Matrix.zeros((n, n))
        for i in range(n):
            matrix[i, i] = 1
        return matrix

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.matrix), len(self.matrix[0]))

    def clone(self) -> Matrix:
        return Matrix(copy.deepcopy(self.matrix))

    def __getitem__(self, key: tuple[int, int]) -> int:
        return self.matrix[key[0]][key[1]]

    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        """
        행렬의 특정 위치의 값을 설정하는 메소드

        Args:
            - key (tuple[int, int]): 행렬의 인데스를 나타내는 튜플
            - value (int): 설정할 값
        """
        self.matrix[key[0]][key[1]] = value

    def __matmul__(self, matrix: Matrix) -> Matrix:
        x, m = self.shape
        m1, y = matrix.shape
        assert m == m1

        result = self.zeros((x, y))

        for i in range(x):
            for j in range(y):
                for k in range(m):
                    result[i, j] += self[i, k] * matrix[k, j]
                    result[i, j] %= self.MOD

        return result

    def __pow__(self, n: int) -> Matrix:
        """
        분할정복을 이용해 행렬의 거듭제곱을 계산하는 메소드

        Args:
            - n (int): 거듭제곱의 지수

        Returns:
            - Matrix: 거듭제곱 결과 행렬
        """

        def divide_and_conquer(self, n: int) -> Matrix:
            if n == 0:
                x = self.shape[0]
                return Matrix.eye(x)
            elif n == 1:
                return self
            else:
                half = divide_and_conquer(self, n//2)      
                if n%2 == 0:
                    return half @ half
                else:
                    return half @ half @ self
                   
        return divide_and_conquer(self, n)

    def __repr__(self) -> str:
        """
        행렬을 문자열로 표현하는 메소드

        Returns:
            - result (str): 문자열로 표현된 행렬
        """
        
        n = self.shape[0]
        result = ""
        for i in range(n):
            for j in range(n):
                result += str(self.matrix[i][j]%self.MOD) + " "
            result += "\n"
        return result