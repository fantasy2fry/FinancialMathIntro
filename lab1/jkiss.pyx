cdef unsigned int x = 123456789
cdef unsigned int y = 987654321
cdef unsigned int z = 43219876
cdef unsigned int c = 6543217

cdef unsigned int JKISS():
    """Generates a random unsigned integer using the JKISS algorithm."""
    cdef unsigned long long t
    global x, y, z, c

    x = 314527869 * x + 1234567
    y ^= (y << 5)
    y ^= (y >> 7)
    y ^= (y << 22)
    t = 4294584393ULL * z + c
    c = t >> 32
    z = t
    return x + y + z

def random():
    """Returns a random number in (0,1) using JKISS."""
    return JKISS() / 4294967296.0
