class Item:
    def __init__(self, key, value):
        self.key = key
        self.value = value


class DictEntry:
    def __init__(self, capacity=100):
        self.items = [None] * capacity
        self.size = 0

    def put(self, key, value):
        for item in self.items:
            if item and item.key == key:
                item.value = value
                return
        else:
            self.items[self.size] = Item(key, value)
            self.size += 1

    def get_item(self, key) -> str:
        for item in self.items:
            if item and item.key == key:
                return item.value
        else:
            raise KeyError
    
    def remove_item(self, key):
        print('remove', key)
        for idx, item in enumerate(self.items):
            if item and item.key == key:
                self.items[idx] = None
                self.size -= 1
                return
        else:
            raise KeyError

    def is_full(self):
        return  self.size >= len(self.items)

    def __contains__(self, key):
        for item in self.items:
            if item and item.key == key:
                return True
        else:
            return False


class Dictionary:
    def __init__(self, dict_capacity=10000, node_capacity=100):
        self.dict_capacity = dict_capacity
        self.node_capacity = node_capacity
        self.num_nodes = int(dict_capacity / node_capacity + 1)
        self.size = 0
        self.nodes = [None] * self.num_nodes

    def __setitem__(self, key, value) -> int:
        index = hash(key) % self.num_nodes
        if not self.nodes[index]:
            self.nodes[index] = DictEntry(capacity=self.node_capacity)
        
        if key in self.nodes[index]:
            self.nodes[index].put(key, value)
        elif self.nodes[index].is_full():
            self.__resize(self.dict_capacity * 4, self.node_capacity * 2)
            self.__setitem__(key, value)
        else:
            self.nodes[index].put(key, value)
            self.size += 1

    def __getitem__(self, key) -> str:
        index = hash(key) % self.num_nodes
        if not self.nodes[index]:
            raise KeyError
        return self.nodes[index].get_item(key)

    def __delitem__(self, key):
        index = hash(key) % self.num_nodes
        if not self.nodes[index]:
            raise KeyError
        self.nodes[index].remove_item(key)
        self.size -= 1
    
    def __contains__(self, key):
        index = hash(key) % self.num_nodes
        if not self.nodes[index]:
            return False
        return key in self.nodes[index]

    def __resize(self, dict_capacity, node_capacity):
        d = Dictionary(dict_capacity=dict_capacity, node_capacity=node_capacity)
        for node in self.nodes:
            if not node:
                continue
            for item in node.items:
                if not item:
                    continue
                d[item.key] = item.value

        self.dict_capacity = d.dict_capacity
        self.node_capacity = d.node_capacity
        self.num_nodes = d.num_nodes
        self.nodes = d.nodes
        print(f'resize to {dict_capacity}')


class BidirectionalDict:
    def __init__(self):
        self.key_index_mapping = Dictionary()
        self.index_key_mapping = Dictionary()
        self.kv = Dictionary()
    
    def __setitem__(self, key, value):
        if key not in self.kv:
            index = self.kv.size
            self.key_index_mapping[key] = index
            self.index_key_mapping[index] = key
        self.kv[key] = value
            
    def __getitem__(self, key):
        return self.kv[key]

    def __delitem__(self, key):
        del self.kv[key]
        index = self.key_index_mapping[key]
        del self.index_key_mapping[index]
        del self.key_index_mapping[key]

    def __contains__(self, key):
        return key in self.kv

    def get_index(self, key):
        return self.key_index_mapping[key]

    def get_key(self, index):
        return self.index_key_mapping[index]
