// 455 Assign Cookies
// 贪心算法，优先供给需求少的小朋友
int findContentChildren(vector<int>& g, vector<int>& s) {
    sort(g.begin(),g.end(),less<int>());
    sort(s.begin(),s.end(),less<int>());

    int res = 0;
    for (int i = 0, j = 0; i < g.size() && j < s.size();) {
        if (g[i] <= s[j]) {
            ++res;
            ++i;
            ++j;
        } else {
            ++j;
        }
    }

    return res;
}

// 135 Candy
// 贪心算法，先保证从左至右满足需求（仅比低优先级多一个糖果）
// 再保证从右向左满足需求，即可
int candy(vector<int>& ratings) {
    if (ratings.size() <= 1) {
        return ratings.size();
    }
     vector<int> res(ratings.size(), 1);
     for(int i = 1; i < ratings.size(); ++i) {
        if (ratings[i] > ratings[i - 1]) {
            res[i] = res[i - 1] + 1;
        }
    }
     for (int i = ratings.size() - 2; i >= 0; --i) {
        if (ratings[i] > ratings[i + 1]) {
            res[i] = max(res[i], res[i + 1] + 1);
        }
    }
     return std::accumulate(res.begin(), res.end(), 0);
}

// 贪心算法，关键在于尽可能选取终点较小的区间（保证尽量多取区间）
// 其后遍历数组即可
int eraseOverlapIntervals(vector<vector<int>>& intervals) {
    if (intervals.size() <= 1) {
        return 0;
    }
    std::sort(intervals.begin(), intervals.end(),[](const vector<int> & left, const vector<int>& right) {
            return left[1] < right[1];
     });

    int res = 0;
    int curLast = intervals[0][1];
    for (int i = 1; i < intervals.size(); ++ i) {
        if (intervals[i][0] < curLast) {
            ++res;
        } else {
            curLast = intervals[i][1];
        }
    }
    return res;
}

// 细节题
// 每一次移位均可确保之前区间已计数完毕,可种植则前进2步，不可种植需要前进三步
// 需要注意末位元素亦需要纳入计数
bool canPlaceFlowers(vector<int>& flowerbed, int n) {
    if (flowerbed.empty()) {
        return false;
    }
    if (n == 0) {
        return true;
    }
    for(int i = 0; i < flowerbed.size(); i += 2) {
        if (flowerbed[i] == 0) {
            if (flowerbed[i + 1] == 0 || i == flowerbed.size()) {
                --n;
                if (n == 0) {
                    return true;
                }
            } else {
                ++i;
            }
        }
    }
     return false;
}

// 很类似的题目，同样是贪心，优先击中小区间内容
// 维护curRight
int findMinArrowShots(vector<vector<int>>& points) {
    if (points.size() <= 1) {
        return points.size();
    }

    std::sort(points.begin(), points.end(), [](const vector<int>& left,const vector<int>& right){
                return left[1] < right[1];
    });

    int res = 1;
    int curRight = points[0][1];
    for (int i = 1; i < points.size(); ++i) {
        if (points[i][0] > curRight) {
            ++res;
            curRight = points[i][1];
        }
    }
    return res;
}

// 贪心算法，核心在于记录每个字符最后一次出现的位置
// 在遍历子串时不断更新当前子串的预期结束位置即可
vector<int> partitionLabels(string S) {
    vector<int> charTables(128, -1);
    for (int i = 0; i < S.size(); ++i) {
        charTables[S[i]] = i;
    }

    vector<int> res;
    int start = 0; // 当前区间起点
    int end = -1; // 当前区间终点
    for (int i = 0; i < S.size(); ++i) {
        end = max(end, charTables[S[i]]); // 在遍历过程中不断更新当前子串预期终点
        if (end == i) {
            res.push_back(end - start + 1);
            start = i + 1;
        }
    }
    return res;
}

// 股市系列的第一题，核心其实是动态规划
// 维持查询最低成本，并一直尝试卖出即可
nt maxProfit(vector<int>& prices) {
    int minValue = INT_MAX; // 最低买入价
    int res = 0;
    for (auto i : prices) {
        minValue = min(minValue, i);
        res = max(res, i - minValue);
    }
    return res;
}

// 股市系列的第二题，核心是贪心
// 思路为每一次都尝试交易，但仅交易能够获利的日子
int maxProfit(vector<int>& prices) {
    int res = 0;
    for (int i = 0; i < prices.size() - 1;++ i) {
        res += max(prices[i + 1] - prices[i], 0);
    }
    return res;
}

// 核心思路在于优先排序，排序规则如注释
vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
    std::sort(people.begin(), people.end(), [](const vector<int>& left, const vector<int>& right){
         // 按照身高执行降序，若身高相同，按排位进行升序
        return left[0] == right[0] ? left[1] < right[1] : left[0] > right[0];
    });

    vector<vector<int>> res;
    for (const auto & p:people) {
        res.insert(res.begin() + p[1], p);
    }

    return res;
}

// 贪心算法的细节题，持续保证前序数组已然升序，若存在逆序对时试图销毁该逆序对
// 需要注意，针对[-1,4,2,2]这种不应该破坏原有升序序列,因此在销毁逆序对时需要取i - 1作判断
bool checkPossibility(vector<int>& nums) {
    if (nums.empty()) {
        return false;
    }

    if (nums.size() <= 2) {
        return true;
    }

    int cnt = 0; // 逆序对数量

    for (int i = 0; i < nums.size() - 1; ++i) {
        if (nums[i] > nums[i + 1]) {
            ++cnt;
            if (cnt == 2) {
                return false;
            }

            // 仅保证在阻塞升序时对后续数字赋值
            if (i != 0 && nums[i -1] > nums[i + 1]) {
                nums[i + 1] = nums[i];
            }
        }
    } 

    return true;
}
