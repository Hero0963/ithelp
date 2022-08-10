from typing import List


class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        matrix = [[0] * n for i in range(n)]

        # set initial values
        total_count = len(matrix) * len(matrix[0])
        count = 1

        directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        d = 0
        boundary = [[0, len(matrix[0]) - 1], [len(matrix) - 1, len(matrix[0]) - 1], [len(matrix) - 1, 0], [1, 0]]
        boundary_renew = [[1, -1], [-1, -1], [-1, 1], [1, 1]]
        hash_map = {}

        i = 0
        j = 0

        while count <= total_count:
            # print(i, j, boundary, ans)
            matrix[i][j] = count
            count += 1

            # check whether touched boundaries
            if [i, j] == boundary[d]:
                # print("touched")
                # update the directions and boundaries
                boundary[d][0] += boundary_renew[d][0]
                boundary[d][1] += boundary_renew[d][1]
                d = (d + 1) % 4

            i, j = i + directions[d][0], j + directions[d][1]

        return matrix
