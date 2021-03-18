// 二分查找求解平方根
// 需要注意l从1起
// 返回值为l-1或r-1
int mySqrt(int x) {
    if (x == 0 || x == 1) {
        return x;
    }
    int l = 1;
    int r = x;
    while (l < r) {
        long mid = l + ((r - l) >> 1);
        long temp = mid * mid;
        if (temp == x) {
            return mid;
        } else if (temp < x) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return r - 1;
}

// 牛顿迭代法
int mySqrt(int x) {
    long res = x;
    // f(x) = x^2 - a;
    // x(n+1) = x(n) - f(xn)/f'(xn)
    // res = res - (res^2 -x)/2*(res)
    while (res * res > x) {
        res = (res + x / res) / 2;
    }
    return res;
}

// lower_bound && upper_bound实现
// 略有出入，并不完全一致
vector<int> searchRange(vector<int>& nums, int target) {
    vector<int> res{-1, -1};

    int l = 0;
    int r = nums.size();
    int mid = 0;

    while (l < r) {
        mid = l + ((r - l) >> 1);
        if (nums[mid] < target) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }

    // l为首个不小于target的元素
    if (l == nums.size() || nums[l] != target) {
        return res;
    } else {
        res[0] = l;
    }

    // 保持l为lower_bound,减小搜索区间
    r = nums.size();
    while (l < r) {
        mid = l + ((r - l) >> 1);
        if (target < nums[mid]) {
            r = mid;
        } else {
            l = mid + 1;
        }
    }

    // l为首个大于target的元素
    res[1] = l - 1;
    return res;
}

// 带有重复元素的旋转数组查找，核心在于若a[mid] ==
// a[l],左移l（此时无法判断是否升序） target与边界作判断时需要使用==
bool search(vector<int>& nums, int target) {
    if (nums.empty()) {
        return false;
    }

    if (nums.size() == 1) {
        return nums[0] == target;
    }

    int l = 0;
    int r = nums.size();
    int mid = 0;

    while (l < r) {
        mid = l + ((r - l) >> 1);
        if (nums[mid] == target) {
            return true;
        }

        if (nums[mid] == nums[l]) {
            ++l;
        } else if (nums[l] < nums[mid]) {  // 左侧升序
            if (nums[l] <= target && target < nums[mid]) {
                r = mid;
            } else {
                l = mid + 1;
            }
        } else {  // 右侧升序
            if (nums[mid] < target && target <= nums[r - 1]) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
    }

    return false;
}

// log(n)级别查找单个元素
// 核心在于异或，若index为奇数，与前一个比较，若想等则说明当前左侧可pass
// 否则与后一个比较
// 可参考特殊case : [2,2,3]
int singleNonDuplicate(vector<int>& nums) {
    if (nums.size() == 1) {
        return nums[0];
    }

    int l = 0;
    int r = nums.size();
    int mid = 0;

    while (l < r) {
        mid = l + ((r - l) >> 1);
        if (nums[mid] == nums[mid ^ 1]) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }

    return nums[r];
}

// 查找中位数及其拓展题
// 注意点主要是pa需要保证<=m
// 比较的是a[pa - 1]
// k从1开始

double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
    int totalSize = nums1.size() + nums2.size();
    if (totalSize & 1) {
        return findKthElementInSortedArrays(nums1.data(), nums1.size(), nums2.data(), nums2.size(), (totalSize >> 1) + 1);
    } else {
        double left = findKthElementInSortedArrays(nums1.data(), nums1.size(), nums2.data(), nums2.size(), (totalSize >> 1));
        double right = findKthElementInSortedArrays(nums1.data(), nums1.size(), nums2.data(), nums2.size(), (totalSize >> 1) + 1);
        return (left + right) / 2.0;
    }    
}

double findKthElementInSortedArrays(int nums1[], int m, int nums2[], int n, int k) {
    if (m > n) { // 保证在较小区间内搜索
        return findKthElementInSortedArrays(nums2, n, nums1, m, k);
    }

    if (m == 0) {
        return nums2[k - 1];
    }

    if (k == 1) {
        return min(nums1[0], nums2[0]);
    }

    int pa = min(m, k >> 1);
    int pb = k - pa;

    if (nums1[pa - 1] == nums2[pb - 1]) {
        return nums1[pa - 1];
    } else if (nums1[pa - 1] > nums2[pb - 1]) {
        return findKthElementInSortedArrays(nums1, m , nums2 + pb, n -pb, k - pb);
    } else {
        return findKthElementInSortedArrays(nums1 + pa, m - pa, nums2, n, k - pa);
    }
}


// https://leetcode.com/problems/search-a-2d-matrix/description/
// 这道题只能暴力遍历逐行，二分求解
// 速度居然还行，击败了80%多
bool searchMatrix(vector<vector<int>>& matrix, int target) {
    if(matrix.empty() || matrix[0].empty()) {
        return false;
    }

    int i = 0;

    while (i < matrix.size()) {
        if (matrix[i][0] == target) {
            return true;
        } else if (matrix[i][0] < target) {
            if (binary_search(matrix[i].begin(), matrix[i].end(), target)){
                return true;
            } else {
                ++i;
            }
        } else {
            return false;
        }
    }
    return false;
}