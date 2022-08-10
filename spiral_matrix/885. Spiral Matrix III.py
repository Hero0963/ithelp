from typing import List


class Solution:
    def spiralMatrixIII(self, rows: int, cols: int, rStart: int, cStart: int) -> List[List[int]]:
        ans = []

        # set initial values
        total_count = rows * cols
        count = 0

        directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        d = 3

        i = rStart
        j = cStart
        path_constraint = 0
        move = 0

        while count < total_count:

            # print(i, j)

            # check whether (i,j) in original matrix
            if 0 <= i < rows and 0 <= j < cols:
                # print("append", i, j)
                ans.append([i, j])
                count += 1

            # check whether to change direction
            if move == path_constraint:
                d = (d + 1) % 4
                move = 0
                if d % 2 == 0:
                    path_constraint += 1

            i, j = i + directions[d][0], j + directions[d][1]
            move += 1

        return ans
