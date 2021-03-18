// 典型的双指针解题思路,没什么需要考虑的
vector<int> twoSum(vector<int>& numbers, int target) {
    int first = 0;
    int last = numbers.size() - 1;

    int sum = 0;
    while (first != last) {
        sum = numbers[first] + numbers[last];
        if (sum == target) {
            return vector<int>{first + 1, last + 1};
        } else if (sum < target) {
            ++first;
        } else {
            --last;
        }
    }

    return vector<int>{-1, -1};
}

// 归并排序，需要注意，因为O(1)空间，因此需要逆序归并，原理类似于memset
void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
    int i = m - 1;
    int j = n - 1;
    int k = m + n - 1;
    while (i >= 0 && j >= 0) {
        if (nums1[i] > nums2[j]) {
            nums1[k--] = nums1[i--];
        } else {
            nums1[k--] = nums2[j--];
        }
    }

    while (j >= 0) {
        nums1[k--] = nums2[j--];
    }
}

// 获取链表环起点，思路是快慢指针，首先找到快慢指针相遇之处
// 记此时距离链表头部步长为k，因为快指针超过慢指针一圈，因此环的长度亦为k
// 假定慢指针此时再走n步可到达环起点，则有k + n = s， s为链表全长度
// 亦知，s - k 必为环起点处（链表全长 - 环长度）
// 因此从链表头部前进，与慢指针相会之处（均走n步）即为环起点处
ListNode* detectCycle(ListNode* head) {
    if (!head || !(head->next)) {
        return nullptr;
    }

    auto p = head;
    auto q = head;
    do {
        p = p->next;
        q = q->next ? q->next->next : nullptr;
    } while (q && p != q);

    if (!q) {
        return nullptr;
    }

    q = head;
    while (p != q) {
        p = p->next;
        q = q->next;
    }

    return p;
}

// 滑动窗口问题
// 首先需要遍历目标串，获取字符频率
// 其次再遍历文本串，在找到目标串后试图右移左边界，寻找最小串
string minWindow(string s, string t) {
    vector<int> chars(128, 0);
    for (auto c : t) {
        chars[c]++;
    }

    int l = 0;
    int minl = 0;
    int minSize = INT_MAX;
    int cnt = 0;  // 当前匹配成功的字符数
    for (int r = 0; r < s.size(); ++r) {
        if (--chars[s[r]] >= 0) {
            ++cnt;
        }

        while (cnt == t.size()) {
            if (r - l + 1 < minSize) {
                minSize = r - l + 1;
                minl = l;
            }

            if (++chars[s[l]] > 0) {
                --cnt;
            }

            ++l;
        }
    }

    return minSize == INT_MAX ? "" : s.substr(minl, minSize);
}

// 传统双指针解法，需要注意两点
// 1. 边界条件为 l <= r ,case如2
// 2. sum需为long，否则存在溢出风险
bool judgeSquareSum(int c) {
    long l = 0;
    long r = sqrt(c);

    while (l <= r) {
        long sum = l * l + r * r;
        if (sum == c) {
            return true;
        } else if (sum < c) {
            ++l;
        } else {
            --r;
        }
    }

    return false;
}

// 核心思想在于，若当前字符不相等，则试图跳过该字符实现匹配
// 因此需要使用辅助函数
bool isPalindrome(const string& s, int l, int r) {
    if (s.empty() || l > r) {
        return false;
    }
    while (l < r) {
        if (s[l++] != s[r--]) {
            return false;
        }
    }
    return true;
}

bool validPalindrome(string s) {
    if (s.size() == 1) {
        return true;
    }

    int l = 0;
    int r = s.size() - 1;

    while (l < r) {
        if (s[l] != s[r]) {
            return isPalindrome(s, l + 1, r) || isPalindrome(s, l, r - 1);
        } else {
            ++l;
            --r;
        }
    }

    return true;
}

// 本题的核心在于先对原始序列，优先按照长度排序，若其一致则按字典序排序
// 如此则可保证第一个得到的合法串即为本题之解
// 两个字符串匹配函数则较为简单，单纯遍历即可
bool IsMatch(const string& s, const string& t) {
    int i = 0;
    int j = 0;
    while (i < s.size() && j < t.size()) {
        if (s[i] == t[j]) {
            ++i;
            ++j;
        } else {
            ++i;
        }
    }
    return j == t.size();
}

string findLongestWord(string s, vector<string>& d) {
    std::sort(d.begin(), d.end(), [](const string& left, const string& right) {
        return left.size() == right.size() ? left < right
                                           : left.size() > right.size();
    });

    for (const auto& sth : d) {
        if (IsMatch(s, sth)) {
            return sth;
        }
    }
    return "";
}

// 多路分治
// 核心思路是遍历出所有不可能出现在子串内的字符，以此作切分点
// 如果当前无切分点，则为递归基，若存在切分点，则需要将末尾作为最后一个切分点加入
// 此外可增加剪枝操作
int longestSubstring(string s, int k) {
    // 统计字符出现频率
    vector<int> chars(128, 0);
    for (auto c : s) {
        ++chars[c];
    }

    vector<int> splits;
    for (int i = 0; i < s.size(); ++i) {
        if (chars[s[i]] < k) {  // 该位置不可能出现于子串内
            splits.push_back(i);
        }
    }

    if (splits.empty()) {
        // 当前子串内字符均满足条件，可直接返回
        return s.size();
    }

    splits.push_back(s.size());  // 如果当前存在可能，则需要考虑整个串
    int l = 0;                   // 当前左边界
    int res = 0;
    for (auto i : splits) {
        if (i - l > res) {  // 剪枝，但仍需要更新l
            res = max(res, longestSubstring(s.substr(l, i - l), k));
        }
        l = i + 1;  // 更新左边界
    }

    return res;
}

// 3. Longest Substring Without Repeating Characters
// 滑动窗口解题
// 需要注意更新左边界时取max，防止造成滑动窗口边界左移的情况，如bad case"abba"
int lengthOfLongestSubstring(string s) {
    if (s.size() <= 1) {
        return s.size();
    }

    vector<int> curChars(128, -1);

    int res = 0;

    for (int l = 0, r = 0; r < s.size(); ++r) {
        if (curChars[s[r]] != -1) {
            l = max(curChars[s[r]], l);  // 更新左边界
        }
        curChars[s[r]] = r + 1;
        res = max(res, r - l + 1);
    }

    return res;
}

bool isPalindrome(string s) {
    if (s.size() == 1) {
        return true;
    }
    int first = 0;
    int last = s.size() - 1;
    while (first < last) {
      while (!isalnum(s[first]) && first < last) {
        ++first;
      }
      while (!isalnum(s[last]) && first < last) {
        --last;
      }
      if (tolower(s[first]) == tolower(s[last])) {
        ++first;
        --last;
      } else {
        return false;
      }
    }
    return true;
}