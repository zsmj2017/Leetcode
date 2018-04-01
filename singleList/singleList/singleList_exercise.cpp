#include <unordered_set>

struct ListNode {
	int val;
	ListNode *next;
	ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* RemoveElements(ListNode* head, int val) {
	//Time:O(n) Space:O(1)
	if (!head) return head;
	ListNode prehead(-1);
	prehead.next = head;
	ListNode *p = &prehead, *q = head;
	while (q) {
		if (q->val == val) p->next = q->next;
		else p = p->next;
		q = q->next;
	}
	return prehead.next;
}

ListNode* ReverseList_1(ListNode* head) {
	//Iterative
	//1->2->3->nullptr change to nullptr<-1<-2<-3
	//Time:O(n) Space:O(n)
	ListNode *prev = nullptr, *cur = head, *tmp;
	while (cur) {
		tmp = cur->next;
		cur->next = prev;
		prev = cur;
		cur = tmp;
	}
	return prev;
}

ListNode* ReverseList_2(ListNode* head) {
	//Recursive
	//As we all know, for kth node, what we want it to do just (k+1)th node's next = kth node
	//So there are "head->next->next = head"
	//Head's next must be nullptr to avoid making a circle
	//Time:O(n) Space:O(n)
	if (head == nullptr || head->next == nullptr) return head;
	ListNode* p = ReverseList_2(head->next);
	head->next->next = head;
	head->next = nullptr;
	return p;
}

ListNode* AddTwoNumbers(ListNode *l1, ListNode *l2) {
	//Time:O(m+n) Space:O(1)
	ListNode preHead(0); ListNode *p = &preHead;
	int extra = 0;
	while (l1 || l2 || extra) {
		int sum = l1 ? l1->val : 0 + l2 ? l2->val : 0 + extra;
		extra = sum / 10;
		p->next = new ListNode(sum % 10);
		l1 = l1 ? l1->next : l1;
		l2 = l2 ? l2->next : l2;
		p = p->next;
	}
	return preHead.next;
}

ListNode* ReverseLinkedList¢ò_1(ListNode *head, int m, int n) {
	//Imitation of ReverseSingleList's iterative method
	//Start         -1->1->2->3->4    reverse form 1 to 3
	//ENDLOOP       -1<->1<-2 3->4
	//Do sth to make the program correct :-)
	//-1->1<-2 3->4 (with 1->4)
	//-1->1<-2<-3 1->4
	//-1->3->2->1->4  OK!
	//Time:O(n) Space:O(1)
	if (head == nullptr || head->next == nullptr || m == n) return head;
	n -= m;
	ListNode *prehead = new ListNode(-1);
	prehead->next = head;
	ListNode *pre = prehead;
	while (--m) pre = pre->next;
	ListNode *cur = pre->next, *preCur = pre, *temp = nullptr;
	while (n--) {
		temp = cur->next;
		cur->next = preCur;
		preCur = cur;
		cur = temp;
	}
	pre->next->next = cur->next;
	cur->next = preCur;
	pre->next = cur;
	return prehead->next;
}

ListNode* ReverseLinkedList¢ò_2(ListNode *head, int m, int n) {
	//1->2->3->4->5 should be 1->4->3->2->5
	//When we start the loop,pre:1 start:2 then:3
	//We need to move the start node to the end node's situation and this process need n-m loops
	//Each process we want to move cur to the next situation and it must link to its next node's next
	//And its pre must link to its next
	//Start     a-b-c-d-e-f-g    reverse from b to f
	//First     a-c-b-d-e-f-g
	//Second    a-d-c-b-e-f-g
	//Third     a-e-d-c-b-f-g
	//Fourth    a-f-e-d-c-b-g
	//Time:O(n) Space:O(1)
	n -= m;
	ListNode prehead(0);
	prehead.next = head;
	ListNode* pre = &prehead;
	while (--m) pre = pre->next;
	ListNode* cur = pre->next;//cur as the start node 
	while (n--) {
		ListNode *then = cur->next;
		cur->next = then->next;
		then->next = pre->next;
		pre->next = then;
	}
	return prehead.next;
}

ListNode* PartitionList(ListNode *head, int x) {
	//Remember the fundamental principle of partiton a list
	//is to seperate it into two lists and then link them again
	//Time:O(n) Space:O(1)
	ListNode node1(-1), node2(-1);
	ListNode *p = &node1, *q = &node2;
	while (head) {
		if (head->val < x) p = p->next = head;
		else q = q->next = head;
		head = head->next;
	}
	p->next = node2.next;
	q->next = nullptr;
	return node1.next;
}

ListNode* RemoveDuplicatesFromSortedList(ListNode *head) {
	//Time:O(n) Space:O(1)
	if (!head) return head;
	ListNode* tmp = head;
	while (tmp  && tmp->next) {
		if (tmp->next->val == tmp->val) {
			tmp->next = tmp->next->next;
		}
		else tmp = tmp->next;
	}
	return head;
}

ListNode* RemoveDuplicatesFromSortedList¢ò(ListNode *head) {
	//Time:O(n) Space:O(1)
	if (head == nullptr || head->next == nullptr) return head;
	ListNode prehaed(-1); prehaed.next = head;
	ListNode *cur = head, *prev = &prehaed;
	while (cur) {
		while (cur->next && cur->val == cur->next->val) {
			cur = cur->next;//cur is the first element whose value dont equal to its next node
		}
		//All elements between prev and cur are been deleted(there can't be sth equal to cur between cur and prev)
		//So we choose cur as the new prev
		if (prev->next == cur) prev = prev->next;
		//Cur is the latest duplicated element so we skip it
		else prev->next = cur->next;
		cur = cur->next;
	}
	return prehaed.next;
}

ListNode* RotateList(ListNode *head, int k) {
	//Circle the link to make it easy to solve
	//Time:O(n) Space:O(1)
	if (!head || !(head->next)) return head;
	int len = 1;
	ListNode *newH, *tail;
	newH = tail = head;
	//Get the list's size
	while (tail->next) {
		tail = tail->next;
		len++;
	}
	tail->next = head; // circle the link
	if (k %= len){
		//New tail node is the (len-k)th node (1st node is head)
		for (auto i = 0; i<len - k; i++) tail = tail->next;
	}
	//New head is the new tail's next node
	newH = tail->next;
	//Break this cycle
	tail->next = nullptr;
	return newH;
}

ListNode* RemoveNthNodeFromEndOfList(ListNode* head, int n) {
	//Use two points
	//when q arrive at the tail p arrive at the right situation
	//Time:O(n) Space:O(1)
	if (!head) return head;
	ListNode prehead(-1); prehead.next = head;
	ListNode* p = &prehead, *q = head;
	while (--n) q = q->next;
	while (q->next) {
		p = p->next;
		q = q->next;
	}
	p->next = p->next->next;
	return prehead.next;
}

ListNode* SwapNodesInPairs_1(ListNode* head) {
	//Just swap two node(cur && then) in one loop
	//When we don't have a pair,end the loop
	//Time:O(n£© Space:O(1)
	ListNode prehead(-1); prehead.next = head;
	ListNode *pre = &prehead, *cur = head, *then = cur->next;
	while (cur && then) {
		ListNode* temp = then->next;
		pre->next = then;
		cur->next = then->next;
		then->next = cur;
		pre = cur;
		if (!(cur->next)) return prehead.next;
		else cur = cur->next;
		if (!(cur->next)) return prehead.next;
		else then = cur->next;
	}
	return prehead.next;
}

ListNode* SwapNodesInPairs_2(ListNode* head) {
	//use pointer-to-pointer and make a concise solution
	//pp is the adress of pair's first element
	//a is pair's first element b is pair's second element
	//When we swap the pair's node
	//*pp=b so now pair's head change to the second element
	//pp = &(a->next),so pp is the adress of the new pair's first element
	//Time:O(n£© Space:O(1)
	ListNode **pp = &head, *a, *b;
	while ((a = *pp) && (b = a->next)) {
		a->next = b->next;
		b->next = a;
		*pp = b;
		pp = &(a->next);
	}
	return head;
}

ListNode* SwapNodesInPairs_3(ListNode* head) {
	//Recursion version
	//For any pair,we just make it's second node link to the first
	//And the first link to the new pair's new first node
	//so we return the new pair's new first node
	//Time:O(n) Space:O(n)
	if (!head || !(head->next)) return head;
	ListNode * p = head->next;
	head->next = SwapNodesInPairs_3(head->next->next);
	p->next = head;
	return p;
}

ListNode* ReverseNodesInKGroup_1(ListNode *head, int k) {
	//Reverse the group n/k times
	//Time:O(n^2) Space:O(1)
	if (!head || k == 1) return head;
	ListNode prehead(-1);
	prehead.next = head;
	ListNode *cur = head, *then, *pre = &prehead;
	int len = 1;
	while (cur = cur->next) ++len;
	while (len >= k) {
		cur = pre->next;
		then = cur->next;
		for (int i = 1; i != k; ++i) {
			//This process like exercise_2_2_2_2
			cur->next = then->next;
			then->next = pre->next;
			pre->next = then;
			then = cur->next;
		}
		//move pre to the cur as the next group's pre
		pre = cur;
		len -= k;
	}
	return prehead.next;
}

ListNode* ReverseNodesInKGroup_2(ListNode* head, int k) {
	//Recursive version
	//Time:O(n) Space:O(1)
	ListNode* cur = head;
	int count = 0;
	while (cur != nullptr && count++ != k) cur = cur->next;  // find the k+1 node
	if (!cur) return head;
	cur = ReverseNodesInKGroup_2(cur, k);//cur is the reversed group's head(we reverse it from tail to head)
	//reverse current group 
	//1-2-3 4
	//3-2-1-4
	while (count--) {
		ListNode* tmp = head->next;
		head->next = cur;
		cur = head;
		head = tmp;
	}
	head = cur;
	return head;
}

struct RandomListNode {
	int label;
	RandomListNode *next, *random;
	RandomListNode(int x) : label(x), next(nullptr), random(nullptr) {}
};

RandomListNode * CopyListWithRandomPointer(RandomListNode *head) {
	//Step1: copy each node and attach it to the correspond node's rare
	//Step2: copy each node's random, To be careful of copy's random = corr's random's next
	//Step3: sperate the single list into two lists
	//Time:O(n) Space:O(1)
	if (!head) return head;
	RandomListNode *cur = head, *next;
	//copy each node
	while (cur) {
		RandomListNode * copy = new RandomListNode(cur->label);
		next = cur->next;
		cur->next = copy;
		cur = next;
		copy->next = next;
	}
	//copy random pointer
	cur = head;
	while (cur) {
		if (cur->random) cur->next->random = cur->random->next;
		cur = cur->next->next;
	}
	//Seperate the list into two lists
	cur = head; next = cur->next;
	RandomListNode prehead(-1);
	prehead.next = next;
	while (cur) {
		cur->next = next->next;
		if (next->next) next->next = next->next->next;//exists the next node
		else break;
		cur = cur->next;
		next = next->next;
	}
	return prehead.next;
}

bool LinkedListCircle_1(ListNode *head) {
	//Use a hash_set to store all nodes
	//Time:O(n) Space:O(n£©
	using std::unordered_set;
	unordered_set<ListNode*> used;
	if (!head) return false;
	while (head) {
		if (used.find(head) == used.end()) {
			used.insert(head);
			head = head->next;
		}
		else return true;
	}
	return false;
}

bool LinkedListCircle_2(ListNode *head) {
	//Use two points
	//p run two step && q run one step
	//If there is a circle,p must be equal to q after a loop
	if (!head) return false;
	ListNode *p = head, *q = p;
	while (p && p->next) {
		p = p->next->next;
		q = q->next;
		if (p == q) return true;
	}
	return false;
}

ListNode* LinkedListCircle¢ò_1(ListNode *head) {
	//Use a hashset
	//Time:O(n) Space:O(n£©
	using std::unordered_set;
	unordered_set<ListNode*> used;
	if (!head) return nullptr;
	while (head) {
		if (used.find(head) == used.end()) {
			used.insert(head);
			head = head->next;
		}
		else return head;
	}
	return nullptr;
}

ListNode* LinkedListCircle¢ò_2(ListNode *head) {
	//It is easy to understand when p meet q,p run fast than q k nodes
	//k is the length of the circle
	//Assume they meet at ith node,
	//Define the distance between start node and ith node: S
	//Define the distance between circle's start node and ith node:M obviously,K=S+M
	//We confirm that p run S step will come back to the circle's start node
	//So we set a point res run from start node,when res meet p
	//Their situation is the circle's start node becuase they both run S steps
	if (!head) return false;
	ListNode *p = head, *q = p;
	int count = 1;
	bool hasCircle = false;
	while (p && p->next) {
		p = p->next->next;
		q = q->next;
		++count;//count is the length of circle
		if (p == q) {
			hasCircle = true;
			break;
		}
	}
	if (!hasCircle) return nullptr;
	ListNode *res = head;
	while (res != p) {
		res = res->next;
		p = p->next;
	}
	return res;
}

ListNode* ReorderList(ListNode *head) {
	//Step1:Find the middle node,seperate the old list into two lists
	//Step2:Reverse the second list
	//Step3:Merge two lists
	//1->2->3->4->5 1->2->3<-4<-5 1->5->2->4->3
	if (!head || !(head->next) || !(head->next->next)) return;
	ListNode * p = head, *q = head;
	//Find the middle node
	while (q->next && q->next->next) {
		p = p->next;
		q = q->next->next;
	}
	//p as the first list's last node and the nodes behind it should be reversed 
	ListNode * cur = p->next;
	p->next = nullptr;
	while (cur) {
		ListNode * temp = cur->next;
		cur->next = p;
		p = cur;
		cur = temp;
	}
	//Now p as the newlist's head
	cur = head;
	while (cur) {
		ListNode *next = cur->next;
		ListNode *newnext = p->next;
		cur->next = p;
		p->next = next;
		cur = next;
		p = newnext;
	}
}
