public class Main {
    public static void main(String[] args) {
        // 初始化一个treeSet(输入有重复)
        treeSet<Integer> ts = buildSet(new int[]{5,3,2,6,2,1,3,6,8,4,9});
        // 打印treeSet(输出无重复)
        System.out.println("————————treeSet打印————————");
        ts.print();
        System.out.println("节点个数: "+ts.size());

        // 添加一个节点
        ts.add(7);
        System.out.println();
        ts.print();
        System.out.println("节点个数: "+ts.size());

        // 删除一个节点
        ts.remove(5);
        System.out.println();
        ts.print();
        System.out.println("节点个数: "+ts.size());

        // （中序）遍历的Visitor
        // 外部遍历所有元素
        System.out.println("\n————————外部遍历所有元素————————");
        ts.traversal(new treeSet.Visitor<Integer>() {
            int i = 1;
            public boolean visit(Integer element) {
                System.out.println("这是从小到大第" + i + "个元素：\t"+ element);
                i++;
                return false;
            }
        });

        System.out.println("\n是否包含（有错会报错）?");
        test(ts.contains(3));
        System.out.println("\n是否为空（为空会报错）?");
        test(!ts.isEmpty());
        System.out.println("\n————————清空操作————————");
        ts.clear();
        System.out.println("\n是否为空（不为空会报错）?");
        test(ts.isEmpty());
    }

    public static treeSet buildSet(int[] nodes) {
        treeSet<Integer> bst = new treeSet<>();
        for (int node : nodes) {
            bst.add(node);
        }
        return bst;
    }

    // 此处为MJ老师的断言测试代码
    public static void test(boolean v) {
        if (v) return;
        System.err.println(new RuntimeException().getStackTrace()[1]);
    }
}