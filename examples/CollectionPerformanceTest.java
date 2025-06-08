package examples;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Bài tập thực hành: So sánh hiệu năng các Collection
 * 
 * TODO cho học viên:
 * 1. Chạy các test và ghi lại kết quả
 * 2. Thêm test cho TreeMap vs HashMap
 * 3. Test với different data sizes (1K, 10K, 100K, 1M)
 * 4. Implement test cho LinkedHashSet vs HashSet
 */
public class CollectionPerformanceTest {
    private static final int[] TEST_SIZES = {1000, 10000, 100000};
    
    public static void main(String[] args) {
        System.out.println("=== Collection Performance Comparison ===\n");
        
        for (int size : TEST_SIZES) {
            System.out.println("Testing with " + size + " elements:");
            compareListPerformance(size);
            compareMapPerformance(size);
            compareConcurrentCollections(size);
            System.out.println();
        }
    }
    
    /**
     * So sánh ArrayList vs LinkedList
     * Test cases:
     * 1. Add elements to end
     * 2. Add elements to beginning  
     * 3. Random access
     * 4. Remove from middle
     */
    public static void compareListPerformance(int size) {
        System.out.println("--- List Performance ---");
        
        // Test 1: Add to end
        List<Integer> arrayList = new ArrayList<>();
        List<Integer> linkedList = new LinkedList<>();
        
        long start = System.nanoTime();
        for (int i = 0; i < size; i++) {
            arrayList.add(i);
        }
        long arrayListAddTime = System.nanoTime() - start;
        
        start = System.nanoTime();
        for (int i = 0; i < size; i++) {
            linkedList.add(i);
        }
        long linkedListAddTime = System.nanoTime() - start;
        
        System.out.printf("Add to end - ArrayList: %.2f ms, LinkedList: %.2f ms\n",
            arrayListAddTime / 1_000_000.0, linkedListAddTime / 1_000_000.0);
        
        // Test 2: Add to beginning
        arrayList.clear();
        linkedList.clear();
        
        start = System.nanoTime();
        for (int i = 0; i < Math.min(size, 1000); i++) { // Limit for ArrayList
            arrayList.add(0, i);
        }
        long arrayListInsertTime = System.nanoTime() - start;
        
        start = System.nanoTime();
        for (int i = 0; i < Math.min(size, 1000); i++) {
            linkedList.add(0, i);
        }
        long linkedListInsertTime = System.nanoTime() - start;
        
        System.out.printf("Add to beginning - ArrayList: %.2f ms, LinkedList: %.2f ms\n",
            arrayListInsertTime / 1_000_000.0, linkedListInsertTime / 1_000_000.0);
        
        // Test 3: Random access
        // Repopulate lists
        arrayList.clear();
        linkedList.clear();
        for (int i = 0; i < size; i++) {
            arrayList.add(i);
            linkedList.add(i);
        }
        
        Random random = new Random(42); // Fixed seed for reproducibility
        int accessCount = Math.min(1000, size);
        
        start = System.nanoTime();
        for (int i = 0; i < accessCount; i++) {
            int index = random.nextInt(size);
            arrayList.get(index);
        }
        long arrayListAccessTime = System.nanoTime() - start;
        
        random = new Random(42); // Reset with same seed
        start = System.nanoTime();
        for (int i = 0; i < accessCount; i++) {
            int index = random.nextInt(size);
            linkedList.get(index);
        }
        long linkedListAccessTime = System.nanoTime() - start;
        
        System.out.printf("Random access - ArrayList: %.2f ms, LinkedList: %.2f ms\n",
            arrayListAccessTime / 1_000_000.0, linkedListAccessTime / 1_000_000.0);
    }
    
    /**
     * So sánh HashMap vs TreeMap vs LinkedHashMap
     */
    public static void compareMapPerformance(int size) {
        System.out.println("--- Map Performance ---");
        
        Map<Integer, String> hashMap = new HashMap<>();
        Map<Integer, String> treeMap = new TreeMap<>();
        Map<Integer, String> linkedHashMap = new LinkedHashMap<>();
        
        // Test put operations
        long start = System.nanoTime();
        for (int i = 0; i < size; i++) {
            hashMap.put(i, "Value" + i);
        }
        long hashMapPutTime = System.nanoTime() - start;
        
        start = System.nanoTime();
        for (int i = 0; i < size; i++) {
            treeMap.put(i, "Value" + i);
        }
        long treeMapPutTime = System.nanoTime() - start;
        
        start = System.nanoTime();
        for (int i = 0; i < size; i++) {
            linkedHashMap.put(i, "Value" + i);
        }
        long linkedHashMapPutTime = System.nanoTime() - start;
        
        System.out.printf("Put operations - HashMap: %.2f ms, TreeMap: %.2f ms, LinkedHashMap: %.2f ms\n",
            hashMapPutTime / 1_000_000.0, treeMapPutTime / 1_000_000.0, linkedHashMapPutTime / 1_000_000.0);
        
        // Test get operations
        Random random = new Random(42);
        int getCount = Math.min(1000, size);
        
        start = System.nanoTime();
        for (int i = 0; i < getCount; i++) {
            int key = random.nextInt(size);
            hashMap.get(key);
        }
        long hashMapGetTime = System.nanoTime() - start;
        
        random = new Random(42);
        start = System.nanoTime();
        for (int i = 0; i < getCount; i++) {
            int key = random.nextInt(size);
            treeMap.get(key);
        }
        long treeMapGetTime = System.nanoTime() - start;
        
        System.out.printf("Get operations - HashMap: %.2f ms, TreeMap: %.2f ms\n",
            hashMapGetTime / 1_000_000.0, treeMapGetTime / 1_000_000.0);
    }
    
    /**
     * So sánh HashMap vs ConcurrentHashMap trong môi trường đa luồng
     */
    public static void compareConcurrentCollections(int size) {
        System.out.println("--- Concurrent Collections ---");
        
        Map<Integer, String> hashMap = new HashMap<>();
        Map<Integer, String> concurrentHashMap = new ConcurrentHashMap<>();
        
        // Single-threaded performance
        long start = System.nanoTime();
        for (int i = 0; i < size; i++) {
            hashMap.put(i, "Value" + i);
        }
        long hashMapTime = System.nanoTime() - start;
        
        start = System.nanoTime();
        for (int i = 0; i < size; i++) {
            concurrentHashMap.put(i, "Value" + i);
        }
        long concurrentHashMapTime = System.nanoTime() - start;
        
        System.out.printf("Single-threaded put - HashMap: %.2f ms, ConcurrentHashMap: %.2f ms\n",
            hashMapTime / 1_000_000.0, concurrentHashMapTime / 1_000_000.0);
        
        // TODO: Implement multi-threaded test
        // Hint: Create multiple threads that concurrently modify the maps
        // Measure performance and thread safety
    }
    
    /**
     * Demonstrate hash collision
     */
    public static void demonstrateHashCollision() {
        System.out.println("--- Hash Collision Demo ---");
        
        // These strings have the same hashCode
        String str1 = "Aa";
        String str2 = "BB";
        
        System.out.println("String: " + str1 + ", HashCode: " + str1.hashCode());
        System.out.println("String: " + str2 + ", HashCode: " + str2.hashCode());
        System.out.println("Hash collision: " + (str1.hashCode() == str2.hashCode()));
        
        // Test performance with hash collisions
        Map<String, Integer> map = new HashMap<>();
        
        // Add strings that cause collisions
        long start = System.nanoTime();
        for (int i = 0; i < 1000; i++) {
            // Generate strings with same hash code pattern
            String key = generateCollidingString(i);
            map.put(key, i);
        }
        long collisionTime = System.nanoTime() - start;
        
        System.out.printf("Time with hash collisions: %.2f ms\n", collisionTime / 1_000_000.0);
    }
    
    private static String generateCollidingString(int i) {
        // Simple method to generate strings that might collide
        return "Key" + (i % 100); // This will create some collisions
    }
}
