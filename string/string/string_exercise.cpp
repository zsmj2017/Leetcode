#include <string>
#include <algorithm>
#include <unordered_map>
#include <algorithm>

using namespace::std;

bool ValidPalindrome(string s) {
	//Use two points only need one-pass
	//Time:O(n) Space:O(1)
	for (int i = 0, j = s.size() - 1; i < j; i++, j--) {
		while (isalnum(s[i]) == false && i < j) i++;
		while (isalnum(s[j]) == false && i < j) j--;
		if (toupper(s[i]) != toupper(s[j])) return false;
	}
	return true;
}

bool Implement_strStr_1(string haystack, string needle) {
	//Brute Force_1 -> BM algorithm
	//Time:O(mn) Space:O(1)
	if (needle.empty()) return 0;
	int n = haystack.size(), m = needle.size();
	int i = 0, j = 0;;
	for (; i != n - m + 1; ++i) {
		for (int j = 0; j != m; ++j)
			if (haystack[i + j] != needle[j]) break;//if don't match,move pattern string right
		if (j == m ) break;
	}
	return (i==(n-m+1))?-1:i;
}

bool Implement_strStr_2(string haystack, string needle) {
	//Brute Force_2 -> KMP algorithm
	//Time:O(mn) Space:O(1)
	if (needle.empty()) return 0;
	int n = haystack.size(), m = needle.size();
	int i = 0, j = 0;;
	while (i != m && j != n) {
		if (haystack[i] == needle[j]) ++i, ++j;
		else {
			i -= j - 1;j = 0; // Text String come back to the next index, and pattern string reset
		}
	}
	return (i-j>n-m) ? -1 : i-j;
}

vector<int> NextTable(string p) {
	//KMP's nextTable
	int j = 0, m = p.size();
	int t = -1;
	vector<int> next(m, -1);
	next[0] = -1;
	while (j < m - 1) {
		if (t < 0 || p[j] == p[t]) {
			++j, ++t;
			next[j] = (p[j] != p[t]) ? t : next[t];
		}
		else
			t = next[t];
	}
	return next;
}

int KMP(string t, string p) {
	int i = 0, n = t.size();
	int j = 0, m = p.size();
	vector<int> next(NextTable(p));
	while (i < n && j < m) {
		if (j < 0 || t[i] == p[j])
			++i, ++j;
		else
			j = next[j];
	}
	return (i - j>n - m) ? -1 : i - j;
}

int StringToInteger(string &str) {
	//Time:O(n) Space:O(1)
	long base = 0, sign = 1;
	int i = str.find_first_not_of(" ");
	if (i == -1) return 0;
	else if (str[i] == '+' || str[i] == '-') {
		sign = str[i++] == '-' ? -1 : 1;
	}
	while (i != str.size() && isdigit(str[i])) {
		base = base * 10 + str[i++] - '0';
		if (base*sign >= INT_MAX) return INT_MAX;
		if (base*sign <= INT_MIN) return INT_MIN;
	}
	return base*sign;
}

string AddBinary(string a, string b) {
	int n = a.size(), m = b.size();
	int k = n > m ? n + 1 : m + 1;//the result's length
	string res(k, '0');
	int carry = 0;
	char sum = 0;
	int i = n - 1, j = m - 1;
	while (i != -1 && j != -1) {
		sum = a[i--] - '0' + b[j--] - '0' + carry;
		carry = sum >= 2 ? 1 : 0;
		res[--k] = carry ? sum - 2 + '0' : sum + '0';
	}
	if (i == -1) {
		while (k != 0 && j != -1) {
			sum = b[j--] - '0' + carry;
			carry = sum >= 2 ? 1 : 0;
			res[--k] = carry ? sum - 2 + '0' : sum + '0';
		}
	}
	else if (j == -1) {
		while (k != 0 && i != -1) {
			sum = a[i--] - '0' + carry;
			carry = sum >= 2 ? 1 : 0;
			res[--k] = carry ? sum - 2 + '0' : sum + '0';
		}
	}
	res[0] = carry ? '1' : '0';
	if (res[0] == '0') return res.erase(0, 1);
	else return res;
}

int longestPalindrome(string s) {
	//Given a string which consists of lowercase or uppercase letters, 
	//find the length of the longest palindromes that can be built with those letters.
	//Count from 'A' to 'z' 
	//if one char 'c' appears odd times,so we can confirm the length of P shoule be subtract 1
	//Time:O(n) Space:O(1)
	int odds = 0;
	for (char c = 'A'; c <= 'z'; c++)
		//We can store all char appear times
		//Increase space for decreasing time
		odds += count(s.begin(), s.end(), c) & 1;
	//if odds>0 we can insert a odd char into the middle of string
	return s.size() - odds + (odds > 0);
}

int longestPalindromeSubseq(string s) {
	//use DP 
	//dp[i][j]== the longest Palindrome Subsequence's length in substring(s[i],s[j])
	//DP trans transition
	//dp[i][j]=dp[i+1][j-1]+2 if s[i]==s[j]
	//otherwise,dp[i][j]=max(dp[i+1][j],dp[i][j-1])
	//Initialization: dp[i][i] = 1
	//Time:O(n^2) Space:O(n^2)
	vector<vector<int>> dp(s.size(), vector<int>(s.size(), 0));
	for (int i = s.size() - 1; i >= 0; --i) {
		dp[i][i] = 1;
		for (int j = i + 1; j < s.size(); j++) {
			if (s[i] == s[j]) dp[i][j] = dp[i + 1][j - 1] + 2;
			else dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
		}
	}
	return dp[0][s.size() - 1];
}

int expandAroundCenter(const string&s, int left, int right) {
	//return the length of the palindrome string
	while (left >= 0 && right < s.size() && s[left] == s[right]) {
		--left, ++right;
	}
	//when end the loop,s[left]!=s[right]
	return right - left - 1;
}

string LongestPalindromicSubstring_1(string s) {
	//Expand around center
	//There are 2n-1 centers.For example,"abba"
	//each character can be a center and its center are between the two 'b's
	//Time:O(n^2) Space:O(1)
	int start = 0, maxlen = 0;
	for (int i = 0; i != s.size(); ++i) {
		int len1 = expandAroundCenter(s, i, i);
		int len2 = expandAroundCenter(s, i, i + 1);
		int len = max(len1, len2);
		if (len > maxlen) {
			start = i - (len - 1) / 2;//find the start
			maxlen = len;
		}
	}
	return s.substr(start, maxlen);
}

string LongestPalindromicSubstring_2(string s) {
	//use DP
	//DP trans transition
	//dp[i][j]=(s[i]==s[j]) when j==i+1
	//dp[i][j]=((s[i]==s[j])&&(dp[i+1][j-1])) when j>i+1
	//Initialization: dp[i][i] = 1
	//Time:O(n^2) Space:O(n^2)
	vector<vector<int>> dp(s.size(), vector<int>(s.size(), 0));
	int maxlen = 1, start = 0;
	for (int i = s.size() - 1; i >= 0; --i) {
		dp[i][i] = 1;
		for (int j = i + 1; j < s.size(); ++j) {
			if (s[i] == s[j]) {
				if (dp[i][j] = dp[i + 1][j - 1]) {
					if (j - i + 1 > maxlen) {
						start = i;
						maxlen = j - i + 1;
					}
				}
			}
			else dp[i][j] = 0;
		}
	}
	return s.substr(start, maxlen);
}

bool IsMatchRegular_1(string s, string p) {
	//Recursion Version
	//Time:O(n) Space:O(n)
	if (p.empty()) return s.empty();
	//* must be the 2nd character
	//If the next character is not '*',so the current must be match
	if (p[1] != '*') {//Due to the string implementation in the STL, there will be no pointer out of bounds
		if (p[0] == s[0] || (p[0] == '.' && !s.empty())) {
			//match two string's substring
			return IsMatchRegular_1(s.substr(1), p.substr(1));
		}
		else
			return false;
	}
	else {//the next character is "*"
		  //We may ignore this part of the pattern, 
		  //or delete a matching character in the text.
		  //If we have a match on the remaining strings after any of these operations,
		  //the initial inputs must matched.
		while (p[0] == s[0] || p[0] == '.' && !s.empty()) {
			//If p[0] match s[0]
			//First,Ignore p[0]p[1],check s and p.sub(2)
			if (IsMatchRegular_1(s, p.substr(2))) return true;
			//delete a matching character in the text
			s = s.substr(1);
		}
		//if p[0] can't match s[0]
		//we can match (s,p.sub(2)) this operation is equivalent to treat {p[0]p[1]} as empty
		return IsMatchRegular_1(s, p.substr(2));
	}
}

bool IsMatchRegular_2(string s, string p) {
	//use DP
	//Time:O(n^2) Space:O(n^2)
	//dp[i][j] == if s[0..i-1] matches p[0..j-1]
	//if p[j - 1] != '*' dp[i][j] = dp[i - 1][j - 1] && s[i - 1] == p[j - 1]
	//if p[j - 1] == '*', denote p[j - 2] with x
	//dp[i][j] is true if any of the following is true
	// 1) "x*"matches empty: dp[i][j - 2]
	// 2) "x*" repeats >= 1 times: s[i - 1] == x && dp[i - 1][j]
	// '.' matches any single character
	int m = s.size(), n = p.size();
	vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));
	dp[0][0] = true;
	for (int i = 1; i <= m; i++)
		dp[i][0] = false;
	// p[0.., j - 3, j - 2, j - 1] matches empty if p[j - 1] is '*' and p[0..j - 3] matches empty
	for (int j = 1; j <= n; j++)
		dp[0][j] = j > 1 && '*' == p[j - 1] && dp[0][j - 2];
	for (int i = 1; i <= m; i++)
		for (int j = 1; j <= n; j++)
			if (p[j - 1] != '*')
				dp[i][j] = dp[i - 1][j - 1] && (s[i - 1] == p[j - 1] || '.' == p[j - 1]);
			else
				// p[0] cannot be '*' so no need to check "j > 1" here
				dp[i][j] = dp[i][j - 2] || (s[i - 1] == p[j - 2] || '.' == p[j - 2]) && dp[i - 1][j];

	return dp[m][n];
}

bool IsMatchWildcard(string s, string p) {
	int  slen = s.size(), plen = p.size(), i, j, iStar = -1, jStar = -1;
	for (i = 0, j = 0; i<slen; ++i, ++j) {
		if (p[j] == '*') { //meet a new '*', update traceback i/j info
			iStar = i;//record i
			jStar = j;//record j
			--i;
		}
		else {
			if (p[j] != s[i] && p[j] != '?') {  // mismatch happens
				if (iStar >= 0) { // met a '*' before, then do traceback
					i = iStar++;
					j = jStar;
				}
				else return false; // otherwise fail
			}
		}
	}
	while (p[j] == '*') ++j;
	return j == plen;
}

string LongestCommonPrefix_1(vector<string> &strs) {
	//Vertical scanning
	if (strs.empty()) return "";
	for (int i = 0; i != strs[0].size(); ++i) {
		for (int j = 1; j != strs.size(); ++j) {
			if (strs[j][i] != strs[0][i])//There will be no pointer out of bounds in STL
				return strs[0].substr(0, i);
		}
	}
	return strs[0];
}

string LongestCommonPrefix_2(vector<string> &strs) {
	//Horizontal scanning
	if (strs.empty()) return "";
	int rb = strs[0].size() - 1;
	for (int i = 1; i != strs.size(); ++i) {
		for (int j = 0; j <= rb; ++j) {
			if (strs[i][j] != strs[0][j]) rb = j - 1;//There will be no pointer out of bounds in STL
		}
	}
	return strs[0].substr(0, rb + 1);
}

string IntegerToRoman_1(int num) {
	//It is a simple solution that when you look at it you just want to say 'WTF'...
	//Time:O(1) Space:O(1)
	vector<string> M = { "", "M", "MM", "MMM" };
	vector<string> C = { "", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM" };
	vector<string> X = { "", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC" };
	vector<string> I = { "", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX" };
	return M[num / 1000] + C[(num % 1000) / 100] + X[(num % 100) / 10] + I[num % 10];
}

string IntegerToRoman_2(int num) {
	//Find out all the units and then store them
	vector<int> values = { 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 };
	vector<string> numerals = { "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I" };
	string res;
	for (int i = 0; i < values.size(); ++i) {
		while (num >= values[i]) {
			num -= values[i];
			res += numerals[i];
		}
	}
	return res;
}

inline int match(char c) {
	switch (c) {
	case 'I':return 1;
	case 'V':return 5;
	case 'X':return 10;
	case 'L':return 50;
	case 'C':return 100;
	case 'D':return 500;
	case 'M':return 1000;
	default:return 0;
	}
}


int RomanToInteger(string s) {
	//Time:O(n) Space:O(1)
	if (s.size() == 0) return 0;
	if (s.size() == 1) return match(s[0]);
	int res = match(s.back());
	for (int i = s.size() - 2; i != -1; --i) {
		int m = match(s[i]);
		int n = match(s[i + 1]);
		res += (m < n) ? -m : m;
	}
	return res;
}

bool isAnagram(string s, string t) {
	//Time:O(n) Space:O(1)
	if (s.size() != t.size()) return false;
	vector<int> charac(127, 0);
	for (auto c : s) ++charac[c];
	for (auto c : t) --charac[c];
	for (auto i : charac) {
		if (i != 0) return false;
	}
	return true;
}

vector<vector<string>> groupAnagrams(vector<string>& strs) {
	unordered_map<string, vector<string>> mp;
	for (string s : strs) {
		string t = s;
		sort(t.begin(), t.end());
		mp[t].push_back(s);
	}
	vector<vector<string>> anagrams;
	for (auto m : mp) {
		vector<string> anagram(m.second.begin(), m.second.end());
		anagrams.push_back(anagram);
	}
	return anagrams;
}

int LengthOfLastWord(string s) {
	//Time:O(n) Space:O(1)
	if (s.empty()) return 0;
	int i = s.size() - 1;
	while (i != -1 && s[i] == ' ') --i;
	if (i == -1) return 0;
	int start = i;
	while (i != -1 && s[i] != ' ') --i;
	return start - i;
}