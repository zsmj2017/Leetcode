#pragma once
#include <list>
#include <unordered_map>

struct CacheNode {
	int key;
	int value;
	CacheNode(int k, int v) :key(k), value(v) {};
};


class LRUCache {
private:
	std::list<CacheNode> cacheList;
	std::unordered_map<int, std::list<CacheNode>::iterator> cacheMap;
	size_t capacity;
public:
	LRUCache(int cap) :capacity(cap) {};
	int get(int);
	void set(int, int);
private:
	LRUCache() {};//Exists no LRUCache without a cap
};
