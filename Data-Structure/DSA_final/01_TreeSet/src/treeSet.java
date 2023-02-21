public class treeSet<E extends Comparable<E>> {
    // 要求放入二叉树的元素的类型E必须实现可比较接口
    // 继承BinaryTreeInfo以绘制二叉树图
    private int size=0;
    private Node<E> root;

    // 每个节点都是Node对象
    private static class Node<E> {
        Node<E> left;
        Node<E> right;
        Node<E> parent;
        E element;
        Node(E element) {
            this.element = element;
        }
        Node(E element, Node<E> parent) {
            this.element = element;
            this.parent = parent;
        }
        Node(E element, Node<E> left, Node<E> right) {
            this.element = element;
            this.left = left;
            this.right = right;
        }

        // 获取节点的度
        int degree() {
            int ret = 0;
            if (left != null) ret++;
            if (right != null) ret++;
            return ret;
        }
    }

    // 元素数量
    public int size() {
        return size;
    }

    // 是否为空
    public boolean isEmpty() {
        return size == 0;
    }

    // 查看元素是否存在
    public boolean contains(E element) {
        return node(element) != null;
    }

    // 清空
    public void clear() {
        root = null;
        size = 0;
    }

    // 非空检测以满足二叉树要求
    private void nullCheck(E element) {
        if (element == null) {
            throw new IllegalArgumentException("元素不能为null");
        }
    }

    // 添加
    public void add(E element) {
        nullCheck(element);

        if (root == null) {
            root = new Node<>(element);
            size = 1;
            return;
        }

        Node<E> parent = root;
        // 当前跟element进行比较的节点
        Node<E> node = root;
        // do-while查找父节点
        int cmp;
        do {
            // 设node为父节点，不断迭代
            parent = node;
            cmp = element.compareTo(node.element);
            if (cmp > 0) {
                node = node.right;
            } else if (cmp < 0) {
                node = node.left;
            } else {
                node.element = element;
                // 相等直接返回，达到去重目的
                return;
            }
        } while (node != null);  // 表明该父节点没有(左/右)子节点，需添加

        // 创建新节点
        Node<E> newNode = new Node<>(element, parent);
        if (cmp > 0) {  // 大的添加到父节点右边
            parent.right = newNode;
        } else {  // 小的添加到父节点左边
            parent.left = newNode;
        }

        // 增加元素数量
        size++;
    }

    // 查找element所在的node
    private Node<E> node(E element) {
        nullCheck(element);
        // 当前跟element进行比较的节点
        Node<E> node = root;
        // do-while查找父节点
        do {
            int cmp = element.compareTo(node.element);
            if (cmp > 0) {
                node = node.right;
            } else if (cmp < 0) {
                node = node.left;
            } else {
                // 找到了对应的节点，直接返回
                return node;
            }
        } while (node != null);

        // 找不到对应的节点
        return null;
    }

    // 删除元素
    public void remove(E element) {
        // 找到对应的节点
        Node<E> node = node(element);
        if (node == null) return;
        // 元素数量减少应放在上一个remove()方法中
        // 否则会导致删除degree为2的节点时运行两次remove(Node<E> node)
        // 导致size减小了2
        size--;
        remove(node);
    }

    private void remove(Node<E> node) {
        // 节点的度
        int degree = node.degree();
        if (degree == 0) {  // degree为0的节点直接删除
            if (node == root) {
                root = null;
            } else if (node == node.parent.left) {
                node.parent.left = null;
            } else {
                node.parent.right = null;
            }
        } else if (degree == 1) {  // degree为1的节点
            Node<E> child = (node.left != null) ? node.left : node.right;  // 找孩子
            if (node == root) {
                root = child;
                root.parent = null;
            } else {
                child.parent = node.parent;
                if (node == node.parent.left) {
                    node.parent.left = child;
                } else {
                    node.parent.right = child;
                }
            }
        } else {  // degree为2的节点
            // 找到前驱节点
            Node<E> predecessor = predecessor(node);
            // 用前驱节点的值覆盖node节点的值
            node.element = predecessor.element;
            // 删除前驱节点
            remove(predecessor);
        }
    }

    // 找到node的前驱节点
    private Node<E> predecessor(Node<E> node) {
        Node<E> cur = node.left;
        if (cur != null) { // 左子节点不为null
            while (cur.right != null) {
                cur = cur.right;
            }
            return cur;
        }

        // 从父节点、祖父节点中寻找前驱节点
        while (node.parent != null && node == node.parent.left) {
            node = node.parent;
        }
        return node.parent;
    }

    // 中序遍历（从小到大）
    public void traversal(Visitor<E> visitor) {
        if (visitor == null) return;
        traversal(root, visitor);
    }

    private void traversal(Node<E> root, Visitor<E> visitor) {
        // 递归的退出条件
        if (root == null || visitor.stop) return;

        // 中序遍历左子树
        traversal(root.left, visitor);
        // 一旦满足visitor中定义的停止条件就return
        if (visitor.stop) return;
        // 访问根节点
        visitor.stop = visitor.visit(root.element);  // 访问后具体干嘛由visit决定
        // 中序遍历右子树
        traversal(root.right, visitor);
    }

    // 外部访问Visitor
    public static abstract class Visitor<E> {
        private boolean stop;
        // 如果返回值是true，就马上停止遍历
        public abstract boolean visit(E element);
    }

    // 按中序遍历打印treeSet
    public void print() {
        System.out.print("(");
        print(root);
        System.out.println(")");
    }

    public void print(Node root) {
        // 递归的退出条件
        if (root == null) return;

        // 中序遍历左子树
        print(root.left);
        // 访问根节点
        System.out.print(root.element + ", ");
        // 中序遍历右子树
        print(root.right);
    }
}