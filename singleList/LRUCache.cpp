#include "LRUCache.h"

int LRUCache::get(int key) {
	if (cacheMap.find(key) == cacheMap.end())
		return -1;
	else {
		//move the element to the list's begin so we determine it as the recently used element
		cacheList.splice(cacheList.begin(), cacheList, cacheMap[key]);
		cacheMap[key] = cacheList.begin();
		return cacheMap[key]->value;
	}
}

void LRUCache::set(int k, int v) {
	if (cacheMap.find(k) == cacheMap.end()) {
		if (cacheList.size() == capacity) {
			//erase the least used lement from both map and list
			cacheMap.erase(cacheList.back().key);
			cacheList.pop_back();
		}
		//insert the CacheNode into the list and map
		cacheList.insert(cacheList.begin(), CacheNode(k, v));
		cacheMap[k] = cacheList.begin();
	}
	else {
		//Update the CacheNode's value and move it to the list's begin
		cacheMap[k]->value = v;
		cacheList.splice(cacheList.begin(), cacheList, cacheMap[k]);
		cacheMap[k] = cacheList.begin();
	}
}