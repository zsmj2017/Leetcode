// 快速排序
// 核心思路：首先将基准点移动到a[l](本示例未随机指定基准点，直接使用了a[l])
// 从基准点右侧开始遍历
// [l]: 基准值p
// [l + 1, last]: <p
// [last + 1, i): >= p
// [i,r):未确定元素
// 遍历完成后last所在位置即为基准点的位置
// 递归遍历两侧元素
void quickSort(vector<int>& a, int l, int r) {
    if (r - l <= 1) {  // 长度小于等于1
        return;
    }
    int last = l;                      // 已就绪位置
    int pivot = a[l];                  // 选择a[l]为基准点
    for (int i = l + 1; i < r; ++i) {  // 基准点无需比较
        if (a[i] < pivot) {
            swap(a[i], a[++last]);
        }
    }
    swap(a[last], a[l]);  // 需要将基准点移动至此
    quickSort(a, l, last);
    quickSort(a, last + 1, r);
}

// 快速选择
// 本质就是若发现此次基准点为指定元素，则直接返回
// 否则去两侧继续快排
int findKthLargest(vector<int>& nums, int k) {
    return aux(nums, 0, nums.size(), k);
}
int aux(vector<int>& nums, int l, int r, int k) {
    if (r - l <= 1) {
        return nums[k - 1];  // k is always valid
    }

    int pivotIndex = l + (rand() % (r - l));
    swap(nums[l], nums[pivotIndex]);
    int last = l;
    for (int i = l + 1; i < r; ++i) {
        if (nums[i] > nums[l]) {
            swap(nums[i], nums[++last]);
        }
    }
    swap(nums[l], nums[last]);
    if (last == k - 1) {
        return nums[last];
    } else if (last < k - 1) {
        return aux(nums, last + 1, r, k);
    } else {
        return aux(nums, l, last, k);
    }
}

// 桶排序
// 核心就是根据频率值建立桶
vector<int> topKFrequent(vector<int>& nums, int k) {
    std::unordered_map<int, int> temp;
    int maxCount = 0;
    for (int i : nums) {
        maxCount = max(maxCount, ++temp[i]);
    }
    vector<vector<int> > buckets(maxCount);
    for (const auto& p : temp) {
        buckets[p.second - 1].push_back(p.first);
    }

    vector<int> res;
    for (int i = buckets.size() - 1; i >= 0; --i) {
        for (int a : buckets[i]) {
            res.push_back(a);
            if (--k == 0) {
                return res;
            }
        }
    }
    return res;
}

// 同样是桶排序
string frequencySort(string s) {
    std::unordered_map<char, int> chars;
    int maxCount = 0;
    for (char c : s) {
        maxCount = max(++chars[c], maxCount);
    }
    vector<string> buckets(maxCount);
    for (const auto& p : chars) {
        buckets[p.second - 1] += std::string(p.second, p.first);
    }
    string res;
    for (auto it = buckets.rbegin(); it != buckets.rend(); ++it) {
        res += *it;
    }
    return res;
}

// 快速选择的变形，具体可见注释
// 时刻牢记区间起点和终点，则不会存在问题
void sortColors(vector<int>& nums) {
    if (nums.size() <= 1) {
        return ;
    }

    // 0:[0,i)
    // 1:[i,j)
    // todo::[j,k]
    // 2:[k + 1,nums.size())
    int i = 0;
    int j = 0;
    int k = nums.size() - 1;

    while (j <= k) {
        if (nums[j] == 0) {
            swap(nums[j++],nums[i++]); // expand i && j
        } else if (nums[j] == 2) {
            swap(nums[j],nums[k--]); // the value is todo, so j should not ++
        } else {
            j++;// expand j
        }
    }
}
