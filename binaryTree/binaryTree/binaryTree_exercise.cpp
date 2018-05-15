#include <vector>
#include <stack>
#include <deque>
#include <functional>
#include <algorithm>

using namespace::std;

struct ListNode {
	int val;
	ListNode *next;
	ListNode(int x) : val(x), next(nullptr) {}
};

struct TreeNode {
	int val;
	TreeNode *left;
	TreeNode *right;
	TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

struct TreeLinkNode {
	int val;
	TreeLinkNode *left, *right, *next;
	TreeLinkNode(int x) : val(x), left(nullptr), right(nullptr), next(nullptr) {}
};

vector<int> BinaryTreePreOrderTraversal_1(TreeNode* root) {
	//Use stack to transform Tail recursion to iterative version
	//Time:O(n) Space:O(n)
	vector<int> res;
	stack<TreeNode*> stn;
	if (!root) return res;
	stn.push(root);
	while (!stn.empty()) {
		root = stn.top();
		res.push_back(stn.top()->val);
		stn.pop();
		if (root->right) stn.push(root->right);
		if (root->left) stn.push(root->left);
	}
	return res;
}

void visitAlongLeftBranch(TreeNode* root, stack<TreeNode*> &stn, vector<int> &res) {
	//visit along left branch and push each right child into the stack
	while (root) {
		res.push_back(root->val);
		stn.push(root->right);
		root = root->left;
	}
}

vector<int> BinaryTreePreOrderTraversal_2(TreeNode* root) {
	//Step1:Visit along left branch from top to the bottom
	//Step2:Visit their right child from bottom to the top
	//Time:O(n) Space:O(n)
	vector<int> res;
	stack<TreeNode*> stn;
	while (true) {
		visitAlongLeftBranch(root, stn, res);
		if (stn.empty()) break;
		root = stn.top();
		stn.pop();
	}
	return res;
}

vector<int> BinaryTreePreOrderTraversal_3(TreeNode* root) {
	//Morris algorithm
	//Time:O(n) Space:O(1)
	vector<int> res;
	TreeNode* cur, *prev;
	cur = root;
	while (cur) {
		if (!(cur->left)) {
			res.push_back(cur->val);
			prev = cur;
			cur = cur->right;
		}
		else {
			prev = cur->left;
			while (prev->right && prev->right != cur)
				prev = prev->right;
			if (!(prev->right)) {
				res.push_back(cur->val);//The only one difference with inorder traversal
				prev->right = cur;
				prev = cur;
				cur = cur->left;
			}
			else {
				prev->right = nullptr;
				cur = cur->right;
			}
		}
	}
	return res;
}

void goAlongLeftBranch(TreeNode* root, stack<TreeNode*> &stn, vector<int> &res) {
	//record all left childs
	while (root) {
		stn.push(root);
		root = root->left;
	}
}

vector<int> BinaryTreeInOrderTraversal_1(TreeNode* root) {
	//Time:O(n) Space:O(n)
	vector<int> res;
	stack<TreeNode*> stn;
	while (true) {
		goAlongLeftBranch(root, stn, res);
		if (stn.empty()) break;
		root = stn.top();
		stn.pop();
		res.push_back(root->val);
		root = root->right;//turn to the right child
	}
	return res;
}

vector<int> BinaryTreeInOrderTraversal_2(TreeNode* root) {
	//Time:O(n) Space:O(n)
	stack<TreeNode*> stn;
	vector<int> res;
	while (true) {
		if (root) {
			stn.push(root);
			root = root->left;
		}
		else if (!stn.empty()) {
			root = stn.top();
			stn.pop();
			res.push_back(root->val);
			root = root->right;
		}
		else break;
	}
	return res;
}

vector<int> BinaryTreeInOrderTraversal_3(TreeNode* root) {
	//Morris algorithm
	//Step 1: Initialize current as root
	//Step 2: While current is not NULL,
	//If current does not have left child(must be someone's root)
	//cout current¡¯s value
	//Go to the right child, current = current.right
	//Else(Have left child)
	//In current's left subtree,  the rightmost node must be the cur's prev
	//If we had fund it,so the cur should be process(the previous nodes are all processed)
	//If not,we make the cur its right child and then turn to process cur's left child
	vector<int> res;
	TreeNode* cur, *prev;
	cur = root;
	while (cur) {
		if (!(cur->left)) {
			res.push_back(cur->val);
			cur = cur->right;
		}
		else {
			prev = cur->left;
			while (prev->right && prev->right != cur)
				prev = prev->right;
			if (!(prev->right)) {
				prev->right = cur;
				cur = cur->left;
			}
			else {
				res.push_back(cur->val);
				prev->right = nullptr;
				prev = cur;
				cur = cur->right;
			}
		}
	}
	return res;
}

vector<int> BinaryTreePostOrderTraversal_1(TreeNode* root) {
	//use stack
	//Time:O(n) Space:O(n)
	vector<int> res;
	stack<TreeNode*> stn;
	TreeNode* cur = root, *pre = nullptr;
	while (cur || !stn.empty()) {
		if (cur) {
			//go to the leftest node
			stn.push(cur);
			cur = cur->left;
		}
		else {
			TreeNode* top = stn.top();//get the lowest and leftest node
			if (top->right == pre || top->right == nullptr) {// don't have right child or right child has been visited
				res.push_back(top->val);
				pre = top;//record the pre node
				stn.pop();
			}
			else {//have right child
				cur = top->right;//turn to the right subtree
			}
		}
	}
	return res;
}

vector<int> BinaryTreePostOrderTraversal_2(TreeNode* root) {
	//Reverse the process of preorder
	if (!root) return vector<int>();
	stack<TreeNode*> stn;
	deque<int> res;//use a deque to restore res because it can insert element at begin in O(1£©
	stn.push(root);
	TreeNode *p = nullptr;
	while (!stn.empty()) {
		p = stn.top();
		stn.pop();
		res.push_front(p->val);
		if (p->left) stn.push(p->left);
		if (p->right) stn.push(p->right);
	}
	return vector<int>(res.begin(), res.end());
}

void reverse(TreeNode* from, TreeNode* to) {
	TreeNode* x = from, *y = from->right, *z;
	if (from == to) return;
	while (x != to) {
		z = y->right;
		y->right = x;
		x = y;
		y = z;
	}
}

void visit_reverse(TreeNode* from, TreeNode* to, function<void(TreeNode*)> &visit) {
	TreeNode* p = to;
	reverse(from, to);
	while (true) {
		visit(p);
		if (p == from) break;
		p = p->right;
	}
	reverse(to, from);
}

vector<int> BinaryTreePostOrderTraversal_3(TreeNode* root) {
	//Morris algorithm
	vector<int> res;
	TreeNode dummy(-1);
	TreeNode *cur, *prev = nullptr;
	function<void(TreeNode*)> visit = [&res](TreeNode* node) {res.push_back(node->val); };//lambda
	dummy.left = root; cur = &dummy;
	while (cur) {
		if (!cur->left) {//don't have a left child
			cur = cur->right;//turn to the right child
		}
		else {
			TreeNode *node = cur->left;
			while (node->right && node->right != cur) {
				node = node->right;//find the pre node in inorder traversal
			}
			if (!node->right) {
				node->right = cur;//make a connect
				prev = cur;
				cur = cur->left;
			}
			else {
				//has been made a connect
				visit_reverse(cur->left, prev, visit);//visit the all nodes between cur and prev in reverse order
				prev->right = nullptr;
				prev = cur;
				cur = cur->right;
			}
		}
	}
	return res;
}

vector<vector<int> > BinaryTreeLevelOrderTraversal_1(TreeNode* root) {
	//use two queues and reverse preorder traversal
	deque<TreeNode*> current, next;
	vector<vector<int> > res;
	if (!root) return res;
	vector<int> level;
	current.push_back(root);
	while (!current.empty()) {
		while (!current.empty()) {
			root = current.front();
			current.pop_front();
			level.push_back(root->val);
			if (root->left) next.push_back(root->left);
			if (root->right) next.push_back(root->right);
		}
		res.push_back(level);
		level.clear();
		swap(next, current);
	}
	return res;
}

void traverse(TreeNode* root, size_t level, vector<vector<int> > &res) {
	if (!root) return;
	if (level > res.size()) res.push_back(vector<int>());//No vector has been stored in res
	res[level - 1].push_back(root->val);
	traverse(root->left, level + 1, res);
	traverse(root->right, level + 1, res);
}

vector<vector<int> > BinaryTreeLevelOrderTraversal_2(TreeNode* root) {
	//Recusion version
	vector<vector<int> > res;
	traverse(root, 1, res);
	return res;
}

void zigZag(TreeNode* root, size_t level, vector<vector<int> > &res) {
	if (!root) return;
	if (level > res.size()) res.push_back(vector<int>());
	if (level & 0x01) res[level - 1].push_back(root->val);
	else res[level - 1].insert(res[level - 1].begin(), root->val);//maybe use a deque should be more faster
	zigZag(root->left, level + 1, res);
	zigZag(root->right, level + 1, res);
}

vector<vector<int> > BinaryTreeZigzagLevelOrderTraversal_1(TreeNode* root) {
	//recursion version
	vector <vector<int> > res;
	zigZag(root, 1, res);
	return res;
}

vector<vector<int> > BinaryTreeZigzagLevelOrderTraversal_2(TreeNode* root) {
	vector<vector<int> > res;
	if (!root) return res;
	deque<TreeNode*> dtn;
	dtn.push_back(root);
	bool leftToRight = true;
	while (!dtn.empty()) {
		int size = dtn.size();
		vector<int> row(size);
		for (int i = 0; i < size; i++) {
			TreeNode* node = dtn.front();
			dtn.pop_front();
			// find position to fill node's value
			int index = (leftToRight) ? i : (size - 1 - i);
			row[index] = node->val;
			if (node->left) dtn.push_back(node->left);
			if (node->right) dtn.push_back(node->right);
		}
		//after this level
		leftToRight = !leftToRight;
		res.push_back(row);
	}
	return res;
}

void inOrderTraverse(TreeNode* &root, vector<TreeNode*> &res) {
	if (!root) return;
	if (root->left) inOrderTraverse(root->left, res);
	res.push_back(root);
	if (root->right) inOrderTraverse(root->right, res);
}

void swapValue(vector<TreeNode*> &vtn) {
	if (vtn.empty() || vtn.size() == 1) return;
	//Bubble Swap
	for (size_t i = 0; i != vtn.size(); ++i) {
		for (size_t j = 0; j != vtn.size() - i - 1; ++j) {
			if (vtn[j]->val > vtn[j + 1]->val) swap(vtn[j]->val, vtn[j + 1]->val);
		}
	}
}

void RecoverBinarySearchTree_1(TreeNode* root) {
	//O(n)
	vector<TreeNode*> store;
	inOrderTraverse(root, store);
	swapValue(store);
}

void detect(pair<TreeNode*, TreeNode*> &broken, TreeNode* prev, TreeNode* curr) {
	if (prev && prev->val > curr->val) {
		if (!broken.first) broken.first = prev;
		broken.second = curr;
	}
}

void RecoverBinarySearchTree_2(TreeNode* root) {
	//Morris
	pair<TreeNode*, TreeNode*> broken;
	TreeNode* prev = nullptr;
	TreeNode* curr = root;
	while (curr) {
		if (!curr->left) {
			detect(broken, prev, curr);
			prev = curr;
			curr = curr->right;
		}
		else {
			auto node = curr->left;
			while (node->right && node->right != curr) {
				node = node->right;
			}
			if (!node->right) {
				node->right = curr;
				curr = curr->left;
			}
			else {
				detect(broken, prev, curr);
				node->right = nullptr;
				prev = curr;
				curr = curr->right;
			}
		}
	}
	swap(broken.first->val, broken.second->val);
}

bool SameTree_1(TreeNode* p, TreeNode* q) {
	//Recursion version
	if (p == nullptr && q == nullptr) return true;
	if (p && q && p->val == q->val)
		return SameTree_1(p->left, q->left) && SameTree_1(p->right, q->right);
	else return false;
}

bool SameTree_2(TreeNode* p, TreeNode* q) {
	stack<TreeNode*> stn;
	stn.push(p);
	stn.push(q);
	while (!stn.empty()) {
		p = stn.top(); stn.pop();
		q = stn.top(); stn.pop();
		if (!p && !q) continue;
		if (!p || !q) return false;
		if (p->val != q->val) return false;
		stn.push(p->left); stn.push(q->left);
		stn.push(p->right); stn.push(q->right);
	}
	return true;
}

bool isSymmetric(TreeNode* p, TreeNode* q) {
	if (!p || !q) return p == q;
	if (p->val == q->val)
		return isSymmetric(p->left, q->right) && isSymmetric(p->right, q->left);
	else return false;
}

bool SymmetricTree_1(TreeNode* p) {
	if (!p) return true;
	return isSymmetric(p->left, p->right);
}

bool SymmetricTree_2(TreeNode* root) {
	if (!root) return true;
	deque<TreeNode*> qtn;
	TreeNode *p = nullptr, *q = nullptr;
	qtn.push_back(root->left);
	qtn.push_back(root->right);
	while (!qtn.empty()) {
		p = qtn.front(); qtn.pop_front();
		q = qtn.front(); qtn.pop_front();
		if ((!p && q) || (p && !q)) return false;
		if (!p && !q) continue;
		if (p->val != q->val) return false;
		qtn.push_back(p->left);
		qtn.push_back(q->right);
		qtn.push_back(p->right);
		qtn.push_back(q->left);
	}
	return true;
}

int deepth(TreeNode* root) {
	if (!root) return 0;
	return max(deepth(root->left), deepth(root->right)) + 1;
}

bool IsBalanced(TreeNode* root) {
	if (!root) return true;
	return abs(deepth(root->left) - deepth(root->right)) <= 1 && IsBalanced(root->left) && IsBalanced(root->right);
}

void FlattenBinaryTreeToLinkedList_1(TreeNode* root) {
	if (!root) return;
	stack<TreeNode*> stn;
	TreeNode* curr = nullptr, *prev = nullptr;
	stn.push(root);
	while (!stn.empty()) {
		curr = stn.top(); stn.pop();
		if (curr->right) stn.push(curr->right);
		if (curr->left) stn.push(curr->left);
		curr->left = nullptr;
		curr->right = nullptr;
		if (prev) prev->right = curr;
		prev = curr;
	}
}

void FlattenBinaryTreeToLinkedList_2(TreeNode* root) {
	while (root) {
		if (root->left && root->right) {
			TreeNode* t = root->left;
			while (t->right)
				t = t->right;//find the prenode in inorder traverse
			t->right = root->right;//make the prenode to be the curr's right child's parent
		}
		if (root->left)
			root->right = root->left;
		root->left = nullptr;
		root = root->right;
	}
}

void PopulatingNextRightPointersInEachNode¢ò_1(TreeLinkNode* root) {
	//process each level
	if (!root) return;
	TreeLinkNode dummy(-1);
	for (auto curr = root, prev = &dummy; curr; curr = curr->next) {
		if (curr->left) {
			prev->next = curr->left;
			prev = prev->next;
		}
		if (curr->right) {
			prev->next = curr->right;
			prev = prev->next;
		}
	}
	PopulatingNextRightPointersInEachNode¢ò_1(dummy.next);
}

void PopulatingNextRightPointersInEachNode¢ò_2(TreeLinkNode* root) {
	while (root) {
		TreeLinkNode* next = nullptr;
		TreeLinkNode* prev = nullptr;
		for (; root; root = root->next) {
			if (!next) next = (root->left) ? root->left : root->right;
			if (root->left) {
				if (prev) prev->next = root->left;
				prev = root->left;
			}
			if (root->right) {
				if (prev) prev->next = root->right;
				prev = root->right;
			}
		}
		root = next;
	}
}

TreeNode* ConstructBinaryFromPreAndInOrder(vector<int>& inorder, vector<int>& postorder) {
	return  CreateBinaryFromPreAndInOrder(inorder, postorder, 0, inorder.size() - 1, 0, postorder.size() - 1);
}

TreeNode* CreateBinaryFromPreAndInOrder(vector<int> &inorder, vector<int> &postorder, int is, int ie, int ps, int pe) {
	if (ps > pe) return nullptr;
	TreeNode* node = new TreeNode(postorder[pe]);
	int pos = distance(inorder.begin(), find(inorder.begin(), inorder.end(), node->val));
	node->left = CreateBinaryFromPreAndInOrder(inorder, postorder, is, pos - 1, ps, pos - 1 + ps - is);
	node->right = CreateBinaryFromPreAndInOrder(inorder, postorder, pos + 1, ie, pe - ie + pos, pe - 1);
	return node;
}

TreeNode* ConstructBinaryFromInAndPostOrder(vector<int>& inorder, vector<int>& pretorder) {
	return CreateBinaryFromInAndPostOrder(inorder, pretorder, 0, inorder.size() - 1, 0, pretorder.size() - 1);
}

TreeNode* CreateBinaryFromInAndPostOrder(vector<int> &inorder, vector<int> &preorder, int is, int ie, int ps, int pe) {
	//maybe have some mistakes
	if (ps > pe) return nullptr;
	TreeNode* node = new TreeNode(preorder[0]);
	int pos = distance(inorder.begin(), find(inorder.begin(), inorder.end(), node->val));
	node->left = CreateBinaryFromInAndPostOrder(inorder, preorder, is, pos - 1, ps + 1, ps + 1 + pos - is);
	node->right = CreateBinaryFromInAndPostOrder(inorder, preorder, pos + 1, ie, pos + 1, pe);
	return node;
}

int NumTrees(int n) {
	//Catalan
	vector<int> dp(n + 1, 0);
	dp[0] = 1, dp[1] = 1;
	for (int i = 2; i <= n; ++i) {
		for (int k = 1; k <= i; ++k)
			dp[i] += dp[k - 1] * NumTrees(i - k);
	}
	return dp.back();
}

vector<TreeNode*> generate(int start, int end) {
	vector<TreeNode*> subTree;
	if (start > end) {
		subTree.push_back(nullptr);
		return subTree;
	}
	for (int k = start; k <= end; ++k) {
		vector<TreeNode*> leftSubs = generate(start, k - 1);
		vector<TreeNode*> rightSubs = generate(k + 1, end);
		for (auto i : leftSubs) {
			for (auto j : rightSubs) {
				TreeNode *node = new TreeNode(k);
				node->left = i;
				node->right = j;
				subTree.push_back(node);
			}
		}
	}
}

vector<TreeNode*> GenerateBSTrees(int n) {
	return generate(1, n);
}

bool isValidBST(TreeNode* root, long min, long max) {
	if (!root) return true;
	if (root->val <= min || root->val >= max) return false;
	return isValidBST(root->left, min, root->val) && isValidBST(root->right, root->val, max);
}

bool ValidateBST(TreeNode* root) {
	return isValidBST(root, LONG_MIN, LONG_MAX);
}

TreeNode* generateBST(vector<int>::iterator beg, vector<int>::iterator end) {
	if (end - beg < 0) return nullptr;
	auto mid = beg + (end - beg) / 2;
	TreeNode* root = new TreeNode(*mid);
	root->left = generateBST(beg, mid - 1);
	root->right = generateBST(mid + 1, end);
	return root;
}

TreeNode* ConvertSortedArrayToBST(vector<int> &nums) {
	return generateBST(nums.begin(), nums.end());
}

TreeNode* ConvertSortedListToBST_1(ListNode* head) {
	//like array
	if (!head) return nullptr;
	return toBST(head, nullptr);
}

TreeNode* toBST(ListNode* head, ListNode* tail) {
	if (head == tail) return nullptr;
	ListNode *slow = head, *fast = head;
	while (fast != tail && fast->next != tail) {//find the mid
		fast = fast->next->next;
		slow = slow->next;
	}
	TreeNode *root = new TreeNode(slow->val);
	root->left = toBST(head, slow);
	root->right = toBST(slow->next, tail);
	return root;
}

TreeNode* ConvertSortedListToBST_2(ListNode* head) {
	//inorder first generate leftchild then gnerate root and then rightchild
	//O(n)
	ListNode *p = head;
	int len = 0;
	while (p) {
		++len;
		p = p->next;
	}
	return convertBST(head, 0, len - 1);
}

TreeNode* convertBST(ListNode* &p, int beg, int end) {//refernce
	if (beg > end) return nullptr;
	int mid = beg + (end - beg) / 2;
	TreeNode* leftChild = convertBST(p, beg, mid);//when we complete constructing the leftchild,p is the mid
	TreeNode* root = new TreeNode(p->val);
	root->left = leftChild;
	p = p->next;
	TreeNode* rightChild = convertBST(p, mid + 1, end);
	root->right = rightChild;
	return root;
}

int minDepthOfBinaryTree(TreeNode* root) {
	if (!root) return 0;
	int left = minDepthOfBinaryTree(root->left);
	int right = minDepthOfBinaryTree(root->right);
	return (left == 0 || right == 0) ? left + right + 1 : min(left, right) + 1;
}

int MaxDepthOfBinaryTree(TreeNode* root) {
	if (!root) return 0;
	return 1 + max(MaxDepthOfBinaryTree(root->right), MaxDepthOfBinaryTree(root->left));
}

bool HasPathSum(TreeNode* root, int sum) {
	if (!root) return false;
	if (!root->left && !root->right) return root->val == sum;
	return HasPathSum(root->left, sum - root->val) || HasPathSum(root->right, sum - root->val);
}

vector<vector<int>> PathSum(TreeNode* root, int sum) {
	vector<vector<int> > paths;
	vector<int> path;
	pathSumhelper(root, sum, path, paths);
	return paths;
}

void pathSumhelper(TreeNode* root, int sum, vector<int> &path, vector<vector<int> > &paths) {
	if (!root) return;
	path.push_back(root->val);
	if (!root->left && !root->right)
		if (root->val == sum)
			paths.push_back(path);
	pathSumhelper(root->left, sum - root->val, path, paths);
	pathSumhelper(root->right, sum - root->val, path, paths);
	path.pop_back();
}

int BinaryTreeMaximumPathSum(TreeNode* root) {
	if (!root) return 0;
	int maxSum = root->val;
	dfsMaxSum(root, maxSum);
	return maxSum;
}

int dfsMaxSum(TreeNode* root, int &maxSum) {
	if (!root) return 0;
	int leftSum = dfsMaxSum(root->left, maxSum);
	int rightSum = dfsMaxSum(root->right, maxSum);
	int sum = root->val;
	if (leftSum > 0) sum += leftSum;
	if (rightSum > 0) sum += rightSum;
	maxSum = max(maxSum, sum);
	return max(root->val, root->val + max(leftSum, rightSum));
}

void PopulatingNextRightPointersInEachNode(TreeLinkNode* root) {
	connectHelper(root, nullptr);
}

void connectHelper(TreeLinkNode* root, TreeLinkNode* nextNode) {
	if (!root) return;
	root->next = nextNode;
	connectHelper(root->left, root->right);
	if (nextNode) {
		root->left->next = root->right;
		root->right->next = nextNode->left;
	}
	else {
		root->left->next = root->right;
		root->right = nullptr;
	}
}

int SumRootToLeafNumbers(TreeNode* root) {
	int sum = 0;
	return sumTree(root, sum);
}

int sumTree(TreeNode* root, int sum) {
	if (!root) return 0;
	if (!root->left && !root->right) return sum * 10 + root->val;
	return sumTree(root->left, sum * 10 + root->val) + sumTree(root->right, sum * 10 + root->val);
}