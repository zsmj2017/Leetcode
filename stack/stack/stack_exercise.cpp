#include <stack>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <functional>
#include <iostream>

using namespace::std;

bool ValidParentheses(string s) {
	//Time:O(n) Space:O(n)
	if (s.size() & 0x01) return false;
	stack<char> record;
	for (size_t i = 0; i != s.size(); ++i) {
		if (s[i] == '(' || s[i] == '[' || s[i] == '{')
			record.push(s[i]);
		else {
			if (record.empty()) return false;
			if (s[i] = ')') {
				if (record.top() == '(') record.pop();
				else return false;
			}
			else if (s[i] = ']') {
				if (record.top() == '[') record.pop();
				else return false;
			}
			else if (s[i] = '}') {
				if (record.top() == '{') record.pop();
				else return false;
			}
		}
	}
	return record.empty();
}

int LongestValidParentheses_1(string s) {
	//Skin the string from begin to the end
	//when we meet '(' we push its index into the stack
	//when we meet '£©' 
	//1) s[stck.top()]=='(' they are a matching pair,and we pop the top
	//2) otherwise,they are not match so we put the ')' index into the stack
	//When we end the loop,all the element in the stack are the unmatched character
	//so the elements between their index should be matched
	//Time:O(n) Space:O(n)
	if (s.empty()) return 0;
	stack<int> record;
	for (int i = 0; i != s.size(); ++i) {
		if (s[i] == '(') record.push(i);
		else {
			if (record.empty() || s[record.top()] != '(')
				record.push(i);
			else record.pop();
		}
	}
	if (record.empty()) return s.size();
	int end = s.size() - 1;
	int maxlen = -1;
	while (!record.empty()) {
		maxlen = ((end - record.top() - 1) > maxlen) ? end - record.top() - 1 : maxlen;
		end = record.top();
		record.pop();
	}
	maxlen = (end > maxlen) ? end : maxlen;
	return (maxlen & 0x01) ? maxlen + 1 : maxlen;//maxlen can't be odds
}

int LongestValidParentheses_2(string s) {
	//Only need one-pass
	//Time:O(n) Space:O(n)
	int max_len = 0, last = -1; // the position of the last ')'
	stack<int> lefts; // keep track of the positions of non-matching '('s
	for (int i = 0; i < s.size(); ++i) {
		if (s[i] == '(') {
			lefts.push(i);
		}
		else {
			if (lefts.empty()) {
				// no matching left
				last = i;
			}
			else {
				// find a matching pair
				lefts.pop();
				if (lefts.empty()) {
					//all '(' has been matched,the elements from(last,i] is valid
					max_len = max(max_len, i - last);
				}
				else {
					//the elements from (top(),i] is valid
					max_len = max(max_len, i - lefts.top());
				}
			}
		}
	}
	return max_len;
}

int LargestRectangleInHistogram(vector<int> &heights) {
	//Similar to Water Store(Vector_exercise)
	//Time:O(n) Space:O(n)
	int n = heights.size();
	if (n == 0) return 0;
	if (n == 1) return heights[0];
	heights.push_back(0);
	n++;
	int res = 0;
	stack<int> s;
	int i = 0, j = 0, h = 0;
	while (i<n) {
		//when pop all elements >= s[i],we push s[i] into the stack
		if (s.empty() || heights[i]>heights[s.top()]) s.push(i++);
		else {
			//heights[i]<heights[s.top()]
			//so we compute the area from [s.top(),i) ,any elements in [s.top(),i) must be >=s[i]
			h = heights[s.top()];
			s.pop();
			j = s.empty() ? -1 : s.top();
			res = max(res, h*(i - j - 1));
		}
	}
	return res;
}

int EvaluateRPN(vector<string> &tokens) {
	//Time:O(n) Space:O(n)
	unordered_map<string, function<int(int, int) > > map = {
		{ "+" , [](int a, int b) { return a + b; } },
		{ "-" , [](int a, int b) { return a - b; } },
		{ "*" , [](int a, int b) { return a * b; } },
		{ "/" , [](int a, int b) { return a / b; } }
	};
	std::stack<int> stack;
	for (string& s : tokens) {
		if (!map.count(s)) {
			stack.push(stoi(s));
		}
		else {
			int op1 = stack.top();
			stack.pop();
			int op2 = stack.top();
			stack.pop();
			stack.push(map[s](op2, op1));
		}
	}
	return stack.top();
}
