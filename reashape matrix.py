mat = [[1,2,3],[4,5,6]]
r, c = 3, 2

m = len(mat)
n = len(mat[0])

if m * n != r * c:
    print(mat)
else:
    result = [[0]*c for _ in range(r)]
    for i in range(m*n):
        result[i//c][i%c] = mat[i//n][i%n]
    print(result)
