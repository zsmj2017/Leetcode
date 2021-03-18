// 用来存储暂未归类的算法题

// 快速幂算法
double myPow(double x, int n) {
    if (n == 0) {
        return 1.0;
    }

    if (n == 1) {
        return x;
    }

    bool isPositive = false;
    long temp = n;
    if (temp < 0 ) {
        isPositive = true;
        temp = -temp;
    }

    double res = 1.0;
    while (temp) {
        if (temp & 1) {
            res *= x;
        }
        x *= x;
        temp = temp >> 1;
    }
    return isPositive ? 1 / res : res;
}
