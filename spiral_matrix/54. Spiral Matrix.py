from typing import List


class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix:
            return []

        # set initial values
        total_count = len(matrix) * len(matrix[0])
        count = 0

        directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        d = 0
        boundary = [[0, len(matrix[0]) - 1], [len(matrix) - 1, len(matrix[0]) - 1], [len(matrix) - 1, 0], [1, 0]]
        boundary_renew = [[1, -1], [-1, -1], [-1, 1], [1, 1]]
        ans = []

        i = 0
        j = 0

        while count < total_count:
            # print(i, j, boundary, ans)
            element = matrix[i][j]
            ans.append(element)
            count += 1

            # check whether touched boundaries
            if [i, j] == boundary[d]:
                # print("touched")
                # update the directions and boundaries
                boundary[d][0] += boundary_renew[d][0]
                boundary[d][1] += boundary_renew[d][1]
                d = (d + 1) % 4

            i, j = i + directions[d][0], j + directions[d][1]

        return ans


class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:

        t = len(matrix) * len(matrix[0])
        mi, ni = 0, -1
        op = []
        while len(op) != t:

            tmp = matrix[mi]
            if mi == 0:
                for i in tmp:
                    op.append(i)
                del matrix[mi]
                mi = -1
            else:
                for i in tmp[::-1]:
                    op.append(i)
                del matrix[mi]
                mi = 0

            if ni == -1:
                for j in matrix:
                    op.append(j[ni])
                    del j[ni]
                ni = 0
            else:
                for j in matrix[::-1]:
                    op.append(j[ni])
                    del j[ni]
                ni = -1
        return op
