public interface mapMethods<K,V> {

    /**
     * 返回元素数量
     */
    int size();

    /**
     * 是否为空
     */
    boolean isEmpty();

    /**
     * 清空元素
     */
    void clear();

    /**
     * 将key键的值设置为value
     * 无则创建，返回空值；有则改变，返回旧值
     */
    V put(K key, V value);

    /**
     * 获取key键的值
     * @return key键的值
     */
    V get(K key);

    /**
     * 删除key位置的键值对
     * @return 被删除键值对的值
     */
    V remove(K key);

    /**
     * 判断Map中是否包含某个键
     */
    boolean containsKey(K key);

    /**
     * 判断Map中是否包含某个值
     */
    boolean containsValue(V value);

    /**
     * 根据用户提供的visitor进行遍历
     * @param visitor
     */
    void traversal(map.Visitor<K, V> visitor);
}
