// bitwise计算n是否为2的指数
// 该技巧亦可用于计算二进制中1的个数（每一次都去除一个1）
bool isPowerOfTwo(int n) {
    if (n < 1) {
        return false;
    }
    return (n & (n - 1)) == 0;
}