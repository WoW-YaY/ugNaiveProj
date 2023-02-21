public class Main {
    public static void main(String[] args) {
        map<String, Integer> person = new map<>();

        person.put("James",80);
        person.put("Bob",85);
        person.put("John",76);
        person.put("Amy",88);

        System.out.println("————————打印————————");
        person.print();

        // 如果键不存在会添加，并且无返回
        person.put("White",80);
        System.out.println("\n————————添加元素————————");
        person.print();

        // 如果键存在则会覆盖，并返回原值
        System.out.println("\n————————覆盖元素————————");
        System.out.println(person.put("James",90));
        System.out.println(person.put("James",95));
        person.print();

        System.out.println("\nBob的成绩是：" + person.get("Bob"));
        System.out.println("\n是否包含某人（有错会报错）?");
        test(person.containsKey("John"));
        System.out.println("\n是否包含某分（有错会报错）?");
        test(person.containsValue(95));

        System.out.println("\n————————外部遍历所有元素————————");
        // 外部遍历的visitor
        person.traversal(new map.Visitor<String, Integer>() {
            public boolean visit(String key, Integer value) {
                System.out.println(key + "同学的成绩是："+ value);
                return false;
            }
        });

        System.out.println("\n元素个数: "+person.size());
        System.out.println("\n————————删除元素————————");
        person.remove("Amy");
        System.out.println("\n删除后的元素个数: "+person.size());
        System.out.println("\n是否还包含该元素（有错会报错）?");
        test(!person.containsKey("Amy"));

        System.out.println("\n是否不为空（有错会报错）?");
        test(!person.isEmpty());
        System.out.println("\n————————清空————————");
        person.clear();
        System.out.println("\n是否为空（有错会报错）?");
        test(person.isEmpty());
    }

    // 此处为MJ老师的断言测试代码
    public static void test(boolean v) {
        if (v) return;
        System.err.println(new RuntimeException().getStackTrace()[1]);
    }
}
