from matrix import Vector, Matrix


v1 = Vector([[1, 2, 3, 4]])
v2 = Vector([[0, 1, 2, 3]])
mv1 = Matrix([[1, 2, 3, 4]])

m1 = Matrix([[0, 1, 2, 3],
             [4, 5, 6, 7],
             [8, 9, 10, 11],
             ])
m2 = Matrix([[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10, 11, 12],
             ])

m3 = Matrix((2, 5))

# print(m1 + v1)
print(v1 + mv1)

# print(m1 - v1)
print(v1 - mv1)

# print(m1 + 1)
# print(m1 - 1)

# print(m1 + m3)
# print(m1 * m3)

# print(m1 / 0)
# print(0 / m1)

# print(m1 + None)
# print(m1 - None)
# print(m1 * None)
# print(m1 / None)


print(Matrix([[0, 1, 2, 3],
             [4, 5, 6, 7],
             [8, 9, 10, 11],
             ]))

print(Matrix((2, 5)))


# print(Matrix([[0, "lol", 2, 3],
#              [4, 5, 6, 7],
#              [8, 9, 10, 11],
#               ]))

# print(Matrix([[0, 2, 3],
#              [8, 9, 10, 11],
#               ]))

print(m1.T())
print(m1 + m2)

print(m1)
print(m1 * 2)
print(m1 / 2)
print(m1 * m2.T())
print(m1.T().T())

print(v1.dot(v2))

print(mv1 * v1.T())
