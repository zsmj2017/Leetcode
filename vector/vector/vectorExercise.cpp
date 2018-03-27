#include <iostream>
#include <vector>
#include <stack>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <numeric>

using namespace::std;

void RemoveDuplicatesFromSortedArray_1(vector<int> &nums) {
	//Traverse the array to find the each element and then move it to the correct situation
	//Time:O(n) Space:O(1)
	size_t oldsize = nums.size();
	if (oldsize == 0 || oldsize == 1) return;
	size_t newsize = 1;
	for (size_t i = 1; i != oldsize; i++) {
		if (nums[i] != nums[i - 1]) nums[newsize++] = nums[i];
	}
	nums.resize(newsize);
}

void RemoveDuplicatesFromSortedArray_2(vector<int> &nums) {
	//Use STL
	//Time:O(n) Space:O(1)
	auto it = unique(nums.begin(), nums.end());
	nums.erase(it, nums.end());
}

void RemoveDuplicatesFromSortedArray¢ò_1(vector<int> &nums) {
	//Brute Force
	//Time:O(n) Space:O(1£©
	bool flag = false;
	size_t oldsize = nums.size();
	if (oldsize <= 2) return;
	size_t newsize = 1;
	for (size_t i = 0; i != oldsize - 1; i++) {
		//If a[i]==a[i+1] && a[i]!=a[i-1]
		if (nums[i] == nums[i + 1] && flag == false) {
			flag = true;
			newsize++;
			continue;
		}
		else if (nums[i] != nums[i + 1]) {
			nums[newsize++] = nums[i + 1];
			flag = false;
		}
	}
	nums.resize(newsize);
}

void RemoveDuplicatesFromSortedArray¢ò_2(vector<int> &nums,int k) {
	//A general solution for these problems in this type
	//When we meet ith node,we only need to compare a[i] and a[i-k]
	//The elements between them can't influence the result
	size_t oldsize = nums.size();
	if (oldsize <= k) return;
	size_t newsize = k;
	size_t compareIndex = 0;
	for (size_t index = k; index != oldsize; index++) {
		if (nums[index] != nums[compareIndex]) {
			nums[newsize++] = nums[index];
			compareIndex++;//update compare node
		}
	}
	nums.resize(newsize);
}

int SearchInRotatedSortedArray_1(const vector<int> &v, const int target) {
	//First, find min in this array
	//Then binarySearch in [lo,min) or [min,hi)
	//Time:O(logn) Space:O(1)
	if (v.size() == 0) return -1;
	if (v.size() == 1) return v[0] == target ? 0 : -1;
	int lb = 0, rb = v.size() - 1;//range:[lb,rb]
	int mid = 0;
	//find min to confirm left boundary and right boundary
	while (lb <= rb) {
		mid = (lb + rb) >> 1;
		if (v[mid] == target) return mid;
		else if (v[mid] > v[rb]) lb = mid + 1;//exists reverse order pair in (mid,rb],so min exists in this range
		else rb = mid;
	}
	//lb is the min's index
	int minIndex = lb;
	lb = 0, rb = v.size() - 1;//recover lb,rb;
	int lo = (target <= v[rb]) ? minIndex : 0;
	int hi = (lo == minIndex) ? rb : minIndex;
	int realmid = -1;
	//Search [lo,hi)
	while (lo < hi) {
		realmid = (lo + hi) >> 1;
		if (v[realmid] == target) return realmid;
		else if (target < v[realmid]) hi = realmid;
		else lo = realmid + 1;
	}
	return -1;
}

int SearchInRotatedSortedArray_2(const vector<int> &nums, const int target) {
	//Time:O(logn) Space:O(1)
	if (nums.size() == 0) return -1;
	int lo = 0, hi = nums.size();
	int mid = -1;
	while (lo < hi) {
		mid = (lo + hi) >> 1;
		if (nums[mid] == target) return mid;
		if (nums[lo] < nums[mid]) {//[lo, mid] has a right order
			if (nums[lo] <= target && target < nums[mid]) {//taregt in [lo,mid)
				hi = mid;
			}
			else {//target must in (mid,hi)
				lo = mid + 1;
			}
		}
		else {//there is a min in [lo,mid],so [mid,hi) has a right order
			if (nums[mid] < target && target <= nums[hi - 1]) {//taregt in (mid,hi)
				lo = mid + 1;
			}
			else {//target must in [lo,mid)
				hi = mid;
			}
		}
	}
}

bool SearchInRotatedSortedArray¢ò(const vector<int> &nums, const int target) {
	if (nums.size() == 0) return -1;
	int lo = 0, hi = nums.size();
	int mid = -1;
	while (lo < hi) {
		mid = (lo + hi) >> 1;
		if (target == nums[mid]) return true;
		if (nums[lo] < nums[mid]) {//[lo,mid] has a right order
			if (nums[lo] <= target&&target < nums[mid]) {//target in [lo,mid)
				hi = mid;
			}
			else {//target in (mid,hi)
				lo = mid + 1;
			}
		}
		else if (nums[lo] > nums[mid]) {//[lo,mid] has a min,so [mid,hi) has a right order
			if (nums[mid] < target&&target <= nums[hi - 1]) {//taregt in (mid,hi)
				lo = mid + 1;
			}
			else {//target must in [lo,mid)
				hi = mid;
			}
		}
		else {//nums[lo]==nums[mid] can't confirm where mid is,so skip the duplicate
			  //such as {1,2,3,1,1}
			lo++;//lo can't be the target
		}
	}
	return false;
}

int MedianOfTwoSortedArrays(vector<int> &nums1, vector<int> &nums2) {
	//Brute Force
	//Merge Array1 and Array2 into a new Array
	//Time:O(m+n) Space:O(m+n)
	if (nums1.size() == 0 && nums2.size() == 0) return 0;
	int i = 0, j = 0, count = 0;
	vector<int> res(nums1.size() + nums2.size());
	while (i != nums1.size() && j != nums2.size()) 
		res[count++] = (nums1[i] < nums2[j] ? nums1[i++] : nums2[j++]);
	while (i != nums1.size())
		res[count++] = nums1[i++];
	while (j != nums2.size())
		res[count++] = nums2[j++];
	if ((i + j) % 2) return res[(i + j - 1) / 2];
	else return 
		static_cast<double>((res[(i + j) / 2 - 1] + res[(i + j) / 2])) / 2;
}

int find_kth_element(const vector<int> &nums1, const vector<int> &nums2, size_t k) {
	//Just want to improve the previous solution
	//Use a count to find kth element without storing all elements
	//Time:O(m+n) Space:O(1)
	int count = 0, i = 0, j = 0, current = -1;
	while (i != nums1.size() && j != nums2.size()) {
		current = nums1[i] < nums2[j] ? nums1[i++] : nums2[j++];
		if (++count == k) return current;
	}
	if (i != nums1.size()) return nums1[i + k - count - 1];
	else return nums2[j + k - count - 1];
}

typedef  vector<int>::const_iterator VICI;
int find_kth(VICI &Abegin, VICI &Aend, VICI &Bbegin, VICI &Bend, size_t k) {
	//Recursion solution to find kth element in two arrays
	//Time:O(log(m+n)) Space:O(1)
	//Assume A's size <= B's size
	if ((Aend - Abegin) > (Bend - Bbegin))
		return find_kth(Bbegin, Bend, Abegin, Aend, k);
	//recursion base
	if ((Aend - Abegin) == 0) return *(Bbegin + k - 1);
	if (k == 1) return *Abegin > *Bbegin ? *Bbegin : *Abegin;
	//divide k into two parts
	size_t pa = k >> 1 > (Aend - Abegin) ? (Aend - Abegin) : k >> 1;
	size_t pb = k - pa;
	if (*(Abegin + pa - 1) > *(Bbegin + pb - 1))
		return find_kth(Abegin, Aend, Bbegin + pb, Bend, k - pb);
	else if (*(Abegin + pa - 1) < *(Bbegin + pb - 1))
		return find_kth(Abegin + pa, Aend, Bbegin, Bend, k - pa);
	else return *(Abegin + pa - 1);
}

int LongestConsecutiveSequence_1(vector<int> &nums) {
	//use hash_map
	//Time :O(n) Space: O(n)
	if (nums.size() <= 1) return nums.size();
	unordered_map<int, bool> used;
	for_each(nums.cbegin(), nums.cend(), [&used](int i) {used[i] = false; });
	size_t length = 1, maxlength = 0;
	for (auto i : nums) {
		if (used[i]) continue;//skip duplicates
		used[i] = true;
		length = 1;//restore length
		for (int j = i + 1; used.find(j) != used.end(); ++j) {
			used[j] = true;
			++length;
		}
		for (int j = i - 1; used.find(j) != used.end(); --j) {
			used[j] = true;
			++length;
		}
		maxlength = maxlength > length ? maxlength : length;
	}
	return maxlength;
}

int LongestConsecutiveSequence_2(vector<int> &nums) {
	//sort && unique
	//Time: O(n) Space:O(1)
	if (nums.size() <= 1) return nums.size();
	sort(nums.begin(), nums.end());
	nums.erase(unique(nums.begin(), nums.end()), nums.end());
	size_t length = 1, maxlength = 1;
	for (int i = 1; i != nums.size(); i++) {
		if (nums[i] == nums[i - 1] + 1) length++;
		else {
			maxlength = maxlength > length ? maxlength : length;
			length = 1;
		}
	}
	//when ends at the end of the vector,length don't renew the maxlength,so...
	maxlength = maxlength > length ? maxlength : length;
	return maxlength;
}

int LongestConsecutiveSequence_3(vector<int> &nums) {
	// use hashset
	//Time:O(n) Space:O(n)
	if (nums.size() <= 1) return nums.size();
	unordered_set<int> record(nums.cbegin(), nums.cend());
	size_t maxlength = 1;
	for (auto i : nums) {
		if (record.find(i) == record.end()) continue;//had been deleted
		record.erase(i);
		int prev = i - 1, next = i + 1;
		while (record.find(prev) != record.end()) record.erase(prev--);
		while (record.find(next) != record.end()) record.erase(next++);
		maxlength = maxlength > (next - prev - 1) ? maxlength : (next - prev - 1);
	}
	return maxlength;
}

vector<int> TwoSum_1(vector<int> &nums,int target) {
	//Brute Force
	//Time:O(n^2) Space:O(1)
	vector<int> res;
	for (int i = 0; i != nums.size() - 1; ++i) {
		for (int j = i + 1; j != nums.size(); ++j) {
			if (nums[i] + nums[j] == target) {
				res.push_back(i); res.push_back(j);
				return res;
			}
		}
	}
}

vector<int> TwoSum_2(vector<int> &nums, int target) {
	//use Hash_map one-pass
	//Avoid duplicated values's problems
	//Time:O(n) Space:O(n)
	unordered_map<int, size_t> record;
	vector<int> res;
	size_t index = 0;
	for (auto i : nums) {
		if (record.find(target - i) != record.end()) {
			res.push_back(record[target - i]);
			res.push_back(index);
		}
		else
			record[i] = index++;
	}
	return res;
}

vector<vector<int>> ThreeSum_1(vector<int> &nums, int target) {
	//Brute Force
	//Time: O(n^3) Space: O(1)
	vector<vector<int>> record;
	sort(nums.begin(), nums.end());
	for (size_t i = 0; i != nums.size() - 2; ++i) {
		for (size_t j = i + 1; j != nums.size() - 1; ++j) {
			for (size_t k = j + 1; k != nums.size(); ++k) {
				if (nums[i] + nums[j] <= target - nums[j]) {
					if (nums[i] + nums[j] + nums[k] == target) {
						vector<int> res;
						res.push_back(nums[i]);
						res.push_back(nums[j]);
						res.push_back(nums[k]);
						record.push_back(res);
					}
				}
			}
		}
	}
	sort(record.begin(), record.end());
	record.erase(unique(record.begin(), record.end()), record.end());
	return record;
}

vector<vector<int>> ThreeSum_2(vector<int> &nums, int target) {
	//First sort the vector
	//And then move from both sides to the center
	//Time:O(n^2) Space:O(1)
	vector<vector<int>> record;
	if (nums.size() < 3) return record;
	sort(nums.begin(), nums.end());
	for (int index = 0; index != nums.size() - 2; ++index) {
		//Judge the boundary conditions to avoid wasting times
		if (nums[index] + nums[index + 1] + nums[index + 2] > target) break;
		if (nums[index] + nums[nums.size() - 1] + nums[nums.size() - 2] < target) continue;
		int aim = target - nums[index];
		int start = index + 1, end = nums.size() - 1;
		while (start < end) {
			int sum = nums[start] + nums[end];
			if (sum < aim) ++start;
			else if (sum > aim) --end;
			else {
				record.push_back(vector<int>({ nums[index],nums[start],nums[end] }));
				//Skip duplicates of Number 2
				while (start < end && nums[start] == record[record.size() - 1][1]) ++start;
				//Skip duplicates of Number 3
				while (start < end && nums[end] == record[record.size() - 1][2]) --end;
			}
		}
		//Skip duplicates of Number 1
		while ((index != nums.size() - 3) && (nums[index] == nums[index + 1]))
			++index;
	}
}

int ThreeSumClosest(vector<int> &nums, const int target) {
	//Like 3Sum
	//Time: O(n^2) Space: O(1)
	if (nums.size() <= 3) return accumulate(nums.cbegin(), nums.cend(), 0);
	sort(nums.begin(), nums.end());
	int mindis = numeric_limits<int>::max();
	for (int index = 0; index != nums.size() - 2; ++index) {
		int aim = target - nums[index];
		int start = index + 1, end = nums.size() - 1;
		while (start < end) {
			int sum = nums[start] + nums[end];
			int dis = sum - aim;
			mindis = abs(dis) < abs(mindis) ? dis : mindis;
			if (dis<0) ++start;
			else if (dis>0) --end;
			else {
				return target;
			}
		}
	}
	return target + mindis;
}

vector<vector<int>> FourSum(vector<int> &nums, const int target) {
	//We can expand 3Sum solution to a generalized K Sum solution 
	//Time: O(max(nlogn,n^(k-1)) Space:O(1)
	vector<vector<int>> record;
	if (nums.size() < 4) return record;
	sort(nums.begin(), nums.end());
	int n = nums.size();
	for (int i = 0; i != n - 3; ++i) {
		if (nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target) break;
		if (nums[i] + nums[n - 1] + nums[n - 2] + nums[n - 3] < target) continue;
		//3Sum's Solution
		for (int j = i + 1; j != n - 2; ++j) {
			if (nums[j] + nums[j + 1] + nums[j + 2] > target - nums[i]) break;
			if (nums[j] + nums[n - 1] + nums[n - 2] < target - nums[i]) continue;
			int aim = target - nums[i] - nums[j];
			int front = j + 1, back = n - 1;
			while (front < back) {
				int sum = nums[front] + nums[back];
				if (sum < aim) ++front;
				else if (sum > aim) --back;
				else {
					record.push_back(vector<int>({ nums[i],nums[j],nums[front],nums[back] }));
					while (front < back&&nums[front] == record[record.size() - 1][2]) ++front;
					while (front < back&&nums[back] == record[record.size() - 1][3]) --back;
				}
			}
			while ((j != n - 3) && nums[j] == nums[j + 1]) ++j;
		}
		while ((i != n - 4) && nums[i] == nums[i + 1]) ++i;
	}
	return record;
}

int RemoveElement_1(vector<int>& nums, int val) {
	//Time: O(n) Space: O(1)
	if (nums.empty()) return 0;
	int newsize = 0;
	for (size_t i = 0; i != nums.size(); ++i) {
		if (nums[i] != val) nums[newsize++] = nums[i];
	}
	nums.resize(newsize);
	return newsize;
}

int RemoveElement_2(vector<int>& nums, int val) {
	//Use STL Time:O(n) Space: O(1)
	//The 'removed' elements actually don't been removed from the vector
	//Please remember use member fucntion erase after using remove algorithm 
	return remove(nums.begin(), nums.end(), val) - nums.begin();
}

void NextPermutation(vector<int> &nums) {
	//First step: Search the first decreasing element nums[k] from right to left
	//Second step:Search the first element nums[l] which more than nums[k] from right to left
	//Third step:Swap nums[k] and nums[l]
	//Fourth step:Reverse nums from nums[k+1] to nums[n-1] n=nums.size()
	if (nums.size() <= 1) return;
	if (nums.size() == 2) {
		reverse(nums.begin(), nums.end());
		return;
	}
	size_t k = -1;
	for (size_t i = nums.size() - 2; i != -1; --i) {
		if (nums[i] < nums[i + 1]) {
			k = i;
			break;
		}
	}
	if (k == -1) {
		reverse(nums.begin(), nums.end());
		return;
	}
	size_t l = -1;
	for (size_t j = nums.size() - 1; j != k; --j) {
		if (nums[j] > nums[k]) {
			l = j;
			break;
		}
	}
	swap(nums[l], nums[k]);
	reverse(nums.begin() + k + 1, nums.end());
}

string PermutationSequence(int n, int k) {
	//Brute Force && use STL
	string res(n, '0');
	while (n--)
		res[n] = '1' + n;
	while (--k)
		next_permutation(res.begin(), res.end());
	return res;
}

long factorial(size_t n) {
	if (n == 0 || n == 1) return 1;
	else return n*factorial(n - 1);
}

string PermutationSequence(int n, int k) {
	//use Cantor expansion
	//K=a[n]*(n-1)!+a[n-1]*(n-2)!+...+a[i]*(i-1)!+...+a[2]*1!+a[1]*0!
	//Time: O(n^2) Space: O(n)
	string rest(n, '0');
	size_t i = 0;
	for (size_t i = 0; i != n; i++)
		rest[i] = '1' + i;
	//Cantor Coding count from 0 to n!-1
	if (--k == 0) return rest;
	string result(n, '0');
	//decrease the search scope
	auto it = rest.end();
	while (n--)
	{
		result[i] = rest[0] + k / factorial(n);
		it = remove(rest.begin(), it, result[i++]);
		k %= factorial(n);
	}
	return result;
}

bool ValidSudoku(vector<vector<char>>& board) {
	//When meet a number,check whether it had appeared in each row,column,subboard 
	//Time: O(n^2) Space:O(n)
	int row[9][9] = {}, col[9][9] = {}, sub[9][9] = {};
	for (size_t i = 0; i != 9; ++i) {
		for (size_t j = 0; j != 9; ++j) {
			if (board[i][j] == '.') continue;
			else {
				int num = board[i][j] - '1';
				int k = i / 3 * 3 + j / 3;//Kth SubBoard
				if (row[i][num] || col[j][num] || sub[k][num]) return false;
				row[i][num] = 1, col[j][num] = 1, sub[k][num] = 1;
			}
		}
	}
	return true;
}

int restoreWater(vector<int> &height, int i) {
	//Brute Force
	//When meet a element,search its leftmax && rightmax
	//It's water == min(leftmax,rightmax) - it's height unless it's height are more than the level
	//Time: O(n^2) Space: O(1)
	if (i == 0 || i == height.size() - 1) return 0;
	int lmax = 0, rmax = 0;
	if (i == 1) lmax = height[0];
	else {
		for (size_t lindex = i - 1; lindex != -1; --lindex) {
			lmax = height[lindex] > lmax ? height[lindex] : lmax;
		}
	}
	if (i == height.size() - 2) rmax = height[height.size() - 1];
	else {
		for (size_t rindex = i + 1; rindex != height.size(); ++rindex) {
			rmax = height[rindex] > rmax ? height[rindex] : rmax;
		}
	}
	if (lmax <= height[i] || rmax <= height[i]) return 0;
	else return min(lmax, rmax) - height[i];
}

int TrappingRainWater_1(vector<int> &height) {
	if (height.size() <= 2) return 0;
	int sum = 0;
	for (size_t index = 0; index != height.size(); ++index) {
		sum += restoreWater(height, index);
	}
	return sum;
}

int TrappingRainWater_2(vector<int> &height) {
	//Use two points
	//When A[left] <= A[right] we choose A[left] to be the current element
	//And we can affirm that the water only dependant on height of bar in current direction
	//So we calculate the height of the A[left] we only need to consider about the leftmax
	//Time: O(n) Space: O(1)
	if (height.size() <= 2) return 0;
	int left = 0; int right = height.size() - 1;
	int res = 0;
	int maxleft = 0, maxright = 0;
	//When left==right,its means the A[left] is the highest bar
	while (left < right) {
		if (height[left] <= height[right]) {
			if (height[left] >= maxleft) maxleft = height[left];
			else res += maxleft - height[left];
			left++;
		}
		else {
			if (height[right] >= maxright) maxright = height[right];
			else res += maxright - height[right];
			right--;
		}
	}
	return res;
}

int TrappingRainWater_3(vector<int> &height) {
	//Level means min(leftmax,rightmax)
	//As we can see that lower means the current element's height
	//Level must >= lower and level must <= higher(which don't been calculated in current loop) 
	//Remember that there are always a element whose is height more than level
	if (height.size() <= 2) return 0;
	int level = 0, lower = 0, front = 0, back = height.size() - 1, res = 0;
	while (front<back) {
		lower = height[height[front]<height[back] ? front++ : back--];
		level = max(level, lower);
		res += level - lower;
	}
	return res;
}

int TrappingRainWater_4(vector<int> &height) {
	//Use Stack to calculate the water
	//Time: O(n) Space:O(1£©
	if (height.size() <= 2) return 0;
	int ans = 0, current = 0;
	stack<int> st;
	//Traverse the vector
	while (current < height.size()) {
		while (!st.empty() && height[current] > height[st.top()]) {
			//If current's height > top's height so there must be a pit unless there are only 1 element in the stack
			//Process all the elements in the stack whose height <= current's height
			int top = st.top();
			st.pop();
			if (st.empty()) break;//Only 1 element
								  //Now we have 3 elements and they formed a pit
								  //Distance is from 1 to n in this loop,only first time distance==1
			int distance = current - st.top() - 1;
			//top is the lowest element's index
			int bounded_height = min(height[current], height[st.top()]) - height[top];
			ans += distance * bounded_height;
		}
		st.push(current++);
	}
	return ans;
}

void reverseByPD(vector<vector<int>> &matrix) {
	size_t n = matrix.size();
	for (size_t i = 0; i != n; ++i) {
		for (size_t j = 0; j <i; ++j) {
			if (i == j) continue;
			swap(matrix[i][j], matrix[j][i]);
		}
	}
}

void reverseByLA(vector<vector<int>> &matrix) {
	size_t n = matrix.size();
	for (size_t i = 0; i != n; ++i) {
		size_t l = 0, r = n - 1;
		while (l < r) swap(matrix[i][l++], matrix[i][r--]);
	}
}

void RotateImage(vector<vector<int>> &matrix) {
	//Rotate Image
	//Time: O(n^2) Space:O(1)
	if (matrix.size() <= 1) return;
	//Reverse matrix by principal diagonal
	reverseByPD(matrix);
	//Reverse matrix by longitudinal axis
	reverseByLA(matrix);
}

vector<int> plusOne(vector<int>& digits) {
	//High precision addition
	//Time: O(n) Space:O(1)
	int carry = 1;
	vector<int> res(digits);
	for (size_t i = digits.size() - 1; i != -1; --i) {
		res[i] = (digits[i] + carry) % 10;
		carry = (digits[i] + carry) / 10;
	}
	if (res[0] == 0) res.insert(res.begin(), 1);
	return res;
}

int ClimbingStairs_1(int n) {
	//Climbing Stairs is equivalent to getting nth Fibonacci
	//p:F(n-1) q:F(n-2£©
	//Time:O(n) Space:O(1)
	if (n <= 1) return 1;
	int p = 1, q = 1;
	while (--n) {
		q = q + p;
		p = q - p;
	}
	return q;
}

int ClimbingStairs_2(int n) {
	//Use mathematical formula to get the nth Fibonacci
	//Time: O(1) Space:O(1)
	if (n <= 1) return 1;
	double s = sqrt(5);
	return static_cast<int>((pow((1.0 + s) / 2.0, n) - pow((1.0 - s) / 2.0, n)) / s);
}

vector<int> GrayCode(int n) {
	//Gray code can always be converted by binary code
	//Nth Gray Code == n^(n>>1)
	if (n < 1) return vector<int>(1);
	vector<int> res(n, 0);
	for (size_t i = 1; i != n; ++i)
		res[i] = i ^ (i >> 1);
	return res;
}

void SetMatrixZeros_1(vector<vector<int>> &matrix) {
	//Use two array to store status of the matrix's row && col
	//Time:O(mn) Space:O(m+n)
	int m = matrix.size(), n = matrix[0].size();
	if (m == 1 && n == 1) return;
	vector<int> row(m, 1);
	vector<int> col(n, 1);
	for (size_t i = 0; i != m; ++i) {
		for (size_t j = 0; j != n; ++j) {
			if (matrix[i][j] == 0) row[i] = 0, col[j] = 0;
		}
	}
	for (size_t i = 0; i != m; ++i) {
		for (size_t j = 0; j != n; ++j) {
			if (row[i] == 0 || col[j] == 0) matrix[i][j] = 0;
		}
	}
}

void SetMatrixZeros_2(vector<vector<int>> &matrix) {
	//Don't store anything but 0th col's status or 0th row's status
	//So that we don't need to worry about the 0th row's status may be mistakenly recorded
	//If m[i][j]==0 we can make m[i][0]=0 m[0][j]=0
	//So we can conclude that which row or column should be set zero
	//Time:O(mn) space:O(1)
	int m = matrix.size(), n = matrix[0].size();
	int col0 = 1;//0th col's status 
	if (m == 1 && n == 1) return;
	for (size_t i = 0; i != m; ++i) {
		if (matrix[i][0] == 0) col0 = 0;
		//Skip 0th col
		for (size_t j = 1; j != n; ++j) {
			if (matrix[i][j] == 0) matrix[i][0] = matrix[0][j] = 0;
		}
	}
	//We can only choose the direction from a[i][j] to a[0][0] to rewrite the matirx
	//Otherwise the record imforation will be destructed
	for (size_t i = m - 1; i != -1; --i) {
		//Skip 0th col
		for (size_t j = n - 1; j != 0; --j) {
			if (matrix[i][0] == 0 || matrix[0][j] == 0) matrix[i][j] = 0;
		}
		if (col0 == 0) matrix[i][0] = 0;
	}
}

int canTraverse(size_t i, vector<int> & gas, vector<int> &cost) {
	//Determine whether ith node can complete the circuit
	int res = gas[i] - cost[i];
	if (res<0) return -1;
	int n = gas.size();
	//Now arrive at node i+1,and need n-1 loops to return node i
	for (size_t loop = 1; loop != n; ++loop) {
		res += (gas[(i + loop) % n] - cost[(i + loop) % n]);
		if (res < 0) return -1;
	}
	return i;
}

int GasStation_1(vector<int> & gas, vector<int> & cost) {
	//Brute Force
	//Traverse each node to determine whether it can complete the circuit
	//Time:O(n^2) Space:O(1)
	for (size_t index = 0; index != gas.size(); ++index) {
		int res = canTraverse(index, gas, cost);
		if (res != -1) return res;
		else continue;
	}
	return -1;
}

int GasStation_2(vector<int> & gas, vector<int> &cost) {
	//If car starts at A and can not reach B.Any station between [A,B]
	//can not reach B.(B is the first station that A can not reach.)
	//If the total number of gas is bigger than the total number of cost.There must be a solution.
	//The proof can be found in the document "Gas Station.docx"
	int start = 0, total = 0, tank = 0;
	//if car fails at 'start', record the next station
	for (int i = 0; i<gas.size(); i++)
		if ((tank = tank + gas[i] - cost[i])<0) { start = i + 1; total += tank; tank = 0; }
	//If we start from node i and end at node j,so we start at node j+1
	//Total means if we want traverse from i to j+1,we need total gas to overcome the shortage
	return (total + tank < 0) ? -1 : start;
}

int Candy_1(vector<int> &ratings) {
	//Brute Force
	//Always update the canides only when no update occurs in a loop
	//Time: O(n^2) Space:O(n)
	int n = ratings.size();
	if (n <= 1) return n;
	vector<int> candies(n, 1);
	bool flag = true;
	while (flag)
	{
		flag = false;
		for (size_t i = 0; i != n; ++i) {
			if (i != n - 1 && ratings[i] > ratings[i + 1] && candies[i] <= candies[i + 1]) {
				candies[i] = candies[i + 1] + 1;
				flag = true;
			}
			if (i != 0 && ratings[i] > ratings[i - 1] && candies[i] <= candies[i - 1]) {
				candies[i] = candies[i - 1] + 1;
				flag = true;
			}
		}
	}
	return accumulate(candies.begin(), candies.end(), 0);
}

int Candy_2(vector<int> &ratings) {
	//First,we traverse from left to right to make sure that ith node's left element must be right;
	//Second,we traverse from right to left
	//We get the max(left2right,right2left) to satisfy both the left and the right elements
	//Time:O(n) Space:O(1)
	int n = ratings.size();
	if (n <= 1) return n;
	vector<int> left2right(n, 1);
	vector<int> right2left(n, 1);
	for (size_t i = 1; i != n; ++i) {
		if (ratings[i] > ratings[i - 1]) left2right[i] = left2right[i - 1] + 1;
	}
	for (size_t i = n - 2; i != -1; --i) {
		if (ratings[i] > ratings[i + 1]) right2left[i] = right2left[i + 1] + 1;
	}
	int res = 0;
	for (size_t i = 0; i != n; ++i) res += max(left2right[i], right2left[i]);
	return res;
}

int Candy_3(vector<int> &ratings) {
	//Only use one array
	//Time:O(n) Space:O(1)
	//There are also a O(n) O(1) solution, but i don't have time to undersatnd it
	int n = ratings.size();
	if (n <= 1) return n;
	vector<int> res(n, 1);
	for (size_t i = 1; i != n; ++i) {
		if (ratings[i] > ratings[i - 1]) res[i] = res[i - 1] + 1;
	}
	for (size_t i = n - 2; i != -1; --i) {
		//If no need to update 
		if (ratings[i] > ratings[i + 1]) res[i] = max(res[i], res[i + 1] + 1);
	}
	return accumulate(res.begin(), res.end(), 0);
}

int SingleNumber_1(vector<int> &nums) {
	//Not use Bit manipulation
	//Time:O(n) Space:O(1)
	if (nums.size() == 1) return nums[0];
	for (size_t index = 0; index < nums.size() - 2; index += 2) {
		if (nums[index] != nums[index + 1]) {
			if (nums[index + 1] == nums[index + 2]) return nums[index];
			else return nums[index + 1];
		}
	}
}

int SingleNumber_2(vector<int> &nums) {
	//Use Bit manipulation
	//A^A=0   0^A=A
	//Time:O(n) Space:O(1)
	int res = 0;
	for (size_t i = 0; i != nums.size(); ++i) {
		res ^= nums[i];
	}
	return res;
}

int SingleNumber¢ò_1(vector<int> &nums) {
	//Find sum of set bits at ith position in all array elements (no need to consider about '0')
	//The bits with sum not multiple of 3, are the bits of element with single occurrence.
	//Time:O(n) Space:O(1)
	int res = 0;
	int sum, x;
	for (size_t i = 0; i != 32; ++i) {
		sum = 0, x = 1 << i;
		for (size_t j = 0; j != nums.size(); ++j) {
			if (j&x) sum++;
		}
		if (sum % 3) res |= x;
	}
	return res;
}

int SingleNumber¢ò_2(vector<int> &nums) {
	//We know a number appears 3 times at most, so we need 2 bits to store the status
	//We use ones as its first bit and twos as its second bit to store 4 status,and we only use 3 status.
	//In this solution, the loop is 00->10->01->00
	//Now what we need to do is just make ones and twos do what we want them to do.
	//For ¡®ones¡¯, we can get ¡®ones = ones ^ A[i]; if (twos == 1) then ones = 0¡¯, that can be tansformed to ¡®ones = (ones ^ A[i]) & ~twos¡¯.
	//For ¡®twos¡¯, we can get ¡®twos = twos ^ A[i]; if (ones == 1) then twos = 0¡¯ and ¡®twos = (twos ^ A[i]) & ~ones¡¯.
	//Use this method, we can solve any problem with this type
	int ones = 0, twos = 0;
	for (int i = 0; i != nums.size(); i++) {
		ones = (ones ^ nums[i]) & ~twos;
		twos = (twos ^ nums[i]) & ~ones;
	}
	return ones;
}