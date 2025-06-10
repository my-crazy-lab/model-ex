# Java Nâng Cao - Hướng Dẫn Học Tập Chi Tiết

## Lộ Trình Học Tập Được Khuyến Nghị

**Thứ tự ưu tiên học:**
1. **Java Collections Framework** (nền tảng quan trọng)
2. **Generics và Annotations** (cần thiết cho collections nâng cao)
3. **Multithreading và Concurrency** (khó nhất, cần thời gian)
4. **Design Patterns** (áp dụng ngay được)
5. **Unit Testing và TDD** (thực hành song song)
6. **Java I/O và NIO** (ít dùng hơn)
7. **JVM và GC** (nâng cao nhất)
8. **Performance Optimization** (tổng hợp)
9. **Project thực hành** (áp dụng tất cả)

---

## 1. Java Collections Framework Nâng Cao

### 🎯 Mục tiêu học tập
- Hiểu sâu cách hoạt động internal của các collection
- Biết chọn collection phù hợp cho từng tình huống
- Sử dụng thành thạo concurrent collections

### 💻 Code Mẫu và Bài Tập

#### Bài tập 1: So sánh hiệu năng ArrayList vs LinkedList
```java
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class CollectionPerformanceTest {
    private static final int SIZE = 100000;
    
    public static void main(String[] args) {
        compareListPerformance();
        demonstrateHashMapInternals();
        concurrentCollectionExample();
    }
    
    // Bài tập: Đo thời gian thêm/xóa/truy cập phần tử
    public static void compareListPerformance() {
        List<Integer> arrayList = new ArrayList<>();
        List<Integer> linkedList = new LinkedList<>();
        
        // TODO: Implement performance comparison
        // 1. Thêm 100k phần tử vào đầu list
        // 2. Truy cập phần tử ở giữa list
        // 3. Xóa phần tử ở giữa list
        // Đo thời gian và so sánh
    }
    
    // Hiểu cách HashMap hoạt động
    public static void demonstrateHashMapInternals() {
        Map<String, Integer> map = new HashMap<>();
        
        // Tạo hash collision cố ý
        String key1 = "Aa";  // hashCode = 2112
        String key2 = "BB";  // hashCode = 2112 (collision!)
        
        map.put(key1, 1);
        map.put(key2, 2);
        
        System.out.println("Hash collision example:");
        System.out.println(key1 + " hashCode: " + key1.hashCode());
        System.out.println(key2 + " hashCode: " + key2.hashCode());
    }
    
    // Concurrent Collections
    public static void concurrentCollectionExample() {
        ConcurrentHashMap<String, Integer> concurrentMap = new ConcurrentHashMap<>();
        
        // TODO: Tạo nhiều thread cùng modify map
        // So sánh với HashMap thông thường
    }
}
```

### 📝 Quiz Kiểm Tra Kiến Thức

**Câu 1:** ArrayList vs LinkedList - khi nào nên dùng gì?
- A) ArrayList cho random access, LinkedList cho frequent insertion/deletion
- B) LinkedList luôn nhanh hơn ArrayList
- C) ArrayList chỉ dùng cho số nguyên
- D) Không có sự khác biệt

**Câu 2:** HashMap xử lý hash collision như thế nào?
- A) Overwrite giá trị cũ
- B) Sử dụng separate chaining (linked list/tree)
- C) Báo lỗi
- D) Tạo HashMap mới

**Đáp án:** 1-A, 2-B

### 🚀 Cách Cải Thiện Kỹ Năng
1. **Thực hành:** Implement một HashMap đơn giản từ đầu
2. **Đọc source code:** Xem implementation của ArrayList, HashMap trong JDK
3. **Benchmark:** Viết test performance cho các collection khác nhau
4. **Memory analysis:** Dùng VisualVM để xem memory usage của collections

---

## 2. Generics và Annotations

### 🎯 Mục tiêu học tập
- Viết code type-safe với Generics
- Hiểu type erasure và bounded types
- Tạo và sử dụng custom annotations

### 💻 Code Mẫu và Bài Tập

#### Bài tập 2: Generic Repository Pattern
```java
import java.lang.annotation.*;
import java.lang.reflect.Field;
import java.util.*;

// Custom annotation
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.FIELD)
@interface Validate {
    String message() default "Validation failed";
    boolean required() default false;
}

// Generic Repository
public class Repository<T, ID> {
    private Map<ID, T> storage = new HashMap<>();
    
    public void save(T entity) {
        // TODO: Implement validation using reflection
        validateEntity(entity);
        // TODO: Extract ID and save
    }
    
    public Optional<T> findById(ID id) {
        return Optional.ofNullable(storage.get(id));
    }
    
    public List<T> findAll() {
        return new ArrayList<>(storage.values());
    }
    
    // Bounded type example
    public <U extends Comparable<U>> List<T> sortBy(Function<T, U> keyExtractor) {
        // TODO: Implement sorting
        return null;
    }
    
    private void validateEntity(T entity) {
        Class<?> clazz = entity.getClass();
        Field[] fields = clazz.getDeclaredFields();
        
        for (Field field : fields) {
            if (field.isAnnotationPresent(Validate.class)) {
                Validate validation = field.getAnnotation(Validate.class);
                // TODO: Implement validation logic
            }
        }
    }
}

// Example entity
class User {
    @Validate(required = true, message = "Name is required")
    private String name;
    
    @Validate(required = true)
    private String email;
    
    // Constructor, getters, setters
}
```

### 📝 Quiz Kiểm Tra Kiến Thức

**Câu 3:** Type erasure trong Java có nghĩa là gì?
- A) Generic type information bị xóa tại compile time
- B) Generic type information bị xóa tại runtime
- C) Không thể sử dụng generic với primitive types
- D) Generic chỉ hoạt động với String

**Câu 4:** Annotation nào cho phép truy cập tại runtime?
- A) @Retention(RetentionPolicy.SOURCE)
- B) @Retention(RetentionPolicy.CLASS)
- C) @Retention(RetentionPolicy.RUNTIME)
- D) @Retention(RetentionPolicy.COMPILE)

**Đáp án:** 3-B, 4-C

### 🚀 Cách Cải Thiện Kỹ Năng
1. **Thực hành:** Viết generic utility classes (Stack, Queue, Tree)
2. **Framework study:** Xem cách Spring sử dụng annotations
3. **Reflection practice:** Viết annotation processor đơn giản
4. **Type safety:** Thực hành với wildcards (? extends, ? super)

---

## 3. Multithreading và Concurrency

### 🎯 Mục tiêu học tập
- Hiểu thread lifecycle và synchronization
- Sử dụng thành thạo Executor Framework
- Xử lý được race conditions và deadlocks

### 💻 Code Mẫu và Bài Tập

#### Bài tập 3: Producer-Consumer với BlockingQueue
```java
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class ProducerConsumerExample {
    private static final BlockingQueue<String> queue = new ArrayBlockingQueue<>(10);
    private static final AtomicInteger counter = new AtomicInteger(0);
    
    public static void main(String[] args) throws InterruptedException {
        ExecutorService executor = Executors.newFixedThreadPool(4);
        
        // Producer tasks
        for (int i = 0; i < 2; i++) {
            executor.submit(new Producer());
        }
        
        // Consumer tasks
        for (int i = 0; i < 2; i++) {
            executor.submit(new Consumer());
        }
        
        // TODO: Implement graceful shutdown
        Thread.sleep(5000);
        executor.shutdown();
    }
    
    static class Producer implements Runnable {
        @Override
        public void run() {
            try {
                while (!Thread.currentThread().isInterrupted()) {
                    String item = "Item-" + counter.incrementAndGet();
                    queue.put(item);
                    System.out.println("Produced: " + item);
                    Thread.sleep(100);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
    
    static class Consumer implements Runnable {
        @Override
        public void run() {
            try {
                while (!Thread.currentThread().isInterrupted()) {
                    String item = queue.take();
                    System.out.println("Consumed: " + item);
                    Thread.sleep(200);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
}
```

#### Bài tập 4: Deadlock Detection và Prevention
```java
public class DeadlockExample {
    private static final Object lock1 = new Object();
    private static final Object lock2 = new Object();
    
    public static void main(String[] args) {
        // TODO: Tạo deadlock scenario
        // TODO: Implement deadlock prevention strategy
    }
    
    // Bài tập: Viết method để detect deadlock
    public static boolean isDeadlocked() {
        // TODO: Sử dụng ThreadMXBean để detect deadlock
        return false;
    }
}
```

### 📝 Quiz Kiểm Tra Kiến Thức

**Câu 5:** Sự khác biệt giữa synchronized và ReentrantLock?
- A) Không có sự khác biệt
- B) ReentrantLock có thể timeout và interrupt
- C) synchronized nhanh hơn
- D) ReentrantLock chỉ dùng cho static methods

**Đáp án:** 5-B

### 🚀 Cách Cải Thiện Kỹ Năng
1. **Hands-on:** Implement thread-safe data structures
2. **Debugging:** Sử dụng thread dump để analyze deadlocks
3. **Performance:** Benchmark synchronized vs Lock vs Atomic
4. **Real-world:** Implement connection pool với thread safety

---

## 4. Design Patterns trong Java

### 🎯 Mục tiêu học tập
- Áp dụng đúng design patterns trong các tình huống thực tế
- Hiểu trade-offs của từng pattern
- Viết code maintainable và extensible

### 💻 Code Mẫu và Bài Tập

#### Bài tập 5: Implement Observer Pattern cho Event System
```java
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;

// Observer Pattern Implementation
interface EventListener<T> {
    void onEvent(T event);
}

class EventManager<T> {
    private final List<EventListener<T>> listeners = new CopyOnWriteArrayList<>();

    public void subscribe(EventListener<T> listener) {
        listeners.add(listener);
    }

    public void unsubscribe(EventListener<T> listener) {
        listeners.remove(listener);
    }

    public void notify(T event) {
        listeners.forEach(listener -> listener.onEvent(event));
    }
}

// Strategy Pattern Example
interface PaymentStrategy {
    void pay(double amount);
}

class CreditCardPayment implements PaymentStrategy {
    private String cardNumber;

    public CreditCardPayment(String cardNumber) {
        this.cardNumber = cardNumber;
    }

    @Override
    public void pay(double amount) {
        System.out.println("Paid $" + amount + " using Credit Card: " + cardNumber);
    }
}

class PayPalPayment implements PaymentStrategy {
    private String email;

    public PayPalPayment(String email) {
        this.email = email;
    }

    @Override
    public void pay(double amount) {
        System.out.println("Paid $" + amount + " using PayPal: " + email);
    }
}

class ShoppingCart {
    private PaymentStrategy paymentStrategy;
    private double totalAmount;

    public void setPaymentStrategy(PaymentStrategy strategy) {
        this.paymentStrategy = strategy;
    }

    public void checkout() {
        if (paymentStrategy != null) {
            paymentStrategy.pay(totalAmount);
        }
    }

    // TODO: Add items, calculate total, etc.
}
```

#### Bài tập 6: Factory Pattern với Dependency Injection
```java
// Factory Pattern + Dependency Injection
interface DatabaseConnection {
    void connect();
    void disconnect();
}

class MySQLConnection implements DatabaseConnection {
    @Override
    public void connect() {
        System.out.println("Connecting to MySQL...");
    }

    @Override
    public void disconnect() {
        System.out.println("Disconnecting from MySQL...");
    }
}

class PostgreSQLConnection implements DatabaseConnection {
    @Override
    public void connect() {
        System.out.println("Connecting to PostgreSQL...");
    }

    @Override
    public void disconnect() {
        System.out.println("Disconnecting from PostgreSQL...");
    }
}

enum DatabaseType {
    MYSQL, POSTGRESQL, ORACLE
}

class DatabaseConnectionFactory {
    public static DatabaseConnection createConnection(DatabaseType type) {
        switch (type) {
            case MYSQL:
                return new MySQLConnection();
            case POSTGRESQL:
                return new PostgreSQLConnection();
            default:
                throw new IllegalArgumentException("Unsupported database type: " + type);
        }
    }
}

// TODO: Implement Abstract Factory for different database vendors
```

### 📝 Quiz Kiểm Tra Kiến Thức

**Câu 6:** Khi nào nên sử dụng Singleton pattern?
- A) Khi cần đảm bảo chỉ có 1 instance của class
- B) Khi muốn tăng performance
- C) Khi class có nhiều methods
- D) Luôn luôn sử dụng

**Câu 7:** Strategy pattern giải quyết vấn đề gì?
- A) Tạo objects
- B) Thay đổi algorithm tại runtime
- C) Quản lý memory
- D) Thread safety

**Đáp án:** 6-A, 7-B

### 🚀 Cách Cải Thiện Kỹ Năng
1. **Refactoring:** Tìm code cũ và apply design patterns
2. **Framework analysis:** Xem cách Spring, Hibernate sử dụng patterns
3. **Practice:** Implement 23 GoF patterns
4. **Real scenarios:** Áp dụng patterns vào project thực tế

---

## 5. Unit Testing và TDD

### 🎯 Mục tiêu học tập
- Viết test cases hiệu quả với JUnit 5
- Sử dụng Mockito để isolate dependencies
- Thực hành TDD workflow

### 💻 Code Mẫu và Bài Tập

#### Bài tập 7: TDD cho Calculator Service
```java
// Test First - TDD Approach
import org.junit.jupiter.api.*;
import org.mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

// Service to be tested
class CalculatorService {
    private final AuditLogger auditLogger;

    public CalculatorService(AuditLogger auditLogger) {
        this.auditLogger = auditLogger;
    }

    public double divide(double a, double b) {
        if (b == 0) {
            auditLogger.log("Division by zero attempted");
            throw new IllegalArgumentException("Cannot divide by zero");
        }

        double result = a / b;
        auditLogger.log("Division performed: " + a + " / " + b + " = " + result);
        return result;
    }

    // TODO: Implement more operations following TDD
}

interface AuditLogger {
    void log(String message);
}

// Test Class
class CalculatorServiceTest {

    @Mock
    private AuditLogger mockAuditLogger;

    private CalculatorService calculatorService;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        calculatorService = new CalculatorService(mockAuditLogger);
    }

    @Test
    @DisplayName("Should divide two positive numbers correctly")
    void shouldDivideTwoPositiveNumbers() {
        // Given
        double a = 10.0;
        double b = 2.0;
        double expected = 5.0;

        // When
        double result = calculatorService.divide(a, b);

        // Then
        assertEquals(expected, result, 0.001);
        verify(mockAuditLogger).log(contains("Division performed"));
    }

    @Test
    @DisplayName("Should throw exception when dividing by zero")
    void shouldThrowExceptionWhenDividingByZero() {
        // Given
        double a = 10.0;
        double b = 0.0;

        // When & Then
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> calculatorService.divide(a, b)
        );

        assertEquals("Cannot divide by zero", exception.getMessage());
        verify(mockAuditLogger).log("Division by zero attempted");
    }

    // TODO: Write more test cases following TDD
    // 1. Test negative numbers
    // 2. Test very large numbers
    // 3. Test decimal precision
}
```

#### Bài tập 8: Integration Test với TestContainers
```java
// Integration Testing Example
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

@Testcontainers
class UserRepositoryIntegrationTest {

    @Container
    static PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>("postgres:13")
            .withDatabaseName("testdb")
            .withUsername("test")
            .withPassword("test");

    private UserRepository userRepository;

    @BeforeEach
    void setUp() {
        // TODO: Setup database connection and repository
        String jdbcUrl = postgres.getJdbcUrl();
        // Initialize repository with test database
    }

    @Test
    void shouldSaveAndRetrieveUser() {
        // TODO: Implement integration test
        // 1. Save user to database
        // 2. Retrieve user by ID
        // 3. Assert user data is correct
    }
}
```

### 📝 Quiz Kiểm Tra Kiến Thức

**Câu 8:** TDD workflow đúng là gì?
- A) Code -> Test -> Refactor
- B) Test -> Code -> Refactor
- C) Refactor -> Test -> Code
- D) Test -> Refactor -> Code

**Câu 9:** Mock object được sử dụng để?
- A) Tăng performance
- B) Isolate dependencies trong unit test
- C) Thay thế database
- D) Debug code

**Đáp án:** 8-B, 9-B

### 🚀 Cách Cải Thiện Kỹ Năng
1. **Practice TDD:** Viết test trước khi code cho mọi feature mới
2. **Coverage analysis:** Sử dụng JaCoCo để đo test coverage
3. **Test types:** Học unit, integration, end-to-end testing
4. **Continuous testing:** Setup CI/CD với automated testing

---

## 6. Java I/O và NIO

### 🎯 Mục tiêu học tập
- Hiểu sự khác biệt giữa I/O và NIO
- Sử dụng thành thạo Files API
- Xử lý file operations hiệu quả

### 💻 Code Mẫu và Bài Tập

#### Bài tập 9: File Processing với NIO.2
```java
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.io.IOException;
import java.util.stream.Stream;

public class FileProcessingExample {

    public static void main(String[] args) throws IOException {
        Path directory = Paths.get("./test-files");

        // Create test directory and files
        setupTestFiles(directory);

        // Various file operations
        demonstrateFileOperations(directory);

        // File watching
        watchDirectoryChanges(directory);
    }

    private static void setupTestFiles(Path directory) throws IOException {
        Files.createDirectories(directory);

        // Create sample files
        Files.write(directory.resolve("file1.txt"), "Hello World".getBytes());
        Files.write(directory.resolve("file2.txt"), "Java NIO Example".getBytes());
        Files.createDirectory(directory.resolve("subdir"));
    }

    private static void demonstrateFileOperations(Path directory) throws IOException {
        System.out.println("=== File Operations Demo ===");

        // 1. List all files
        try (Stream<Path> files = Files.list(directory)) {
            files.forEach(System.out::println);
        }

        // 2. Walk file tree
        Files.walkFileTree(directory, new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
                System.out.println("Visiting: " + file + " (Size: " + attrs.size() + " bytes)");
                return FileVisitResult.CONTINUE;
            }
        });

        // 3. Find files by pattern
        try (Stream<Path> txtFiles = Files.find(directory, 2,
                (path, attrs) -> path.toString().endsWith(".txt"))) {
            txtFiles.forEach(path -> {
                try {
                    String content = Files.readString(path);
                    System.out.println(path.getFileName() + ": " + content);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            });
        }
    }

    private static void watchDirectoryChanges(Path directory) throws IOException {
        WatchService watchService = FileSystems.getDefault().newWatchService();
        directory.register(watchService,
            StandardWatchEventKinds.ENTRY_CREATE,
            StandardWatchEventKinds.ENTRY_DELETE,
            StandardWatchEventKinds.ENTRY_MODIFY);

        System.out.println("Watching directory: " + directory);

        // TODO: Implement file watching loop
        // Note: This would run indefinitely in real application
    }
}
```

#### Bài tập 10: NIO Channel và Buffer
```java
import java.nio.*;
import java.nio.channels.*;
import java.io.*;

public class NIOChannelExample {

    public static void main(String[] args) throws IOException {
        // File channel example
        copyFileUsingNIO("source.txt", "destination.txt");

        // Memory-mapped file
        readLargeFileEfficiently("large-file.txt");
    }

    private static void copyFileUsingNIO(String source, String destination) throws IOException {
        try (FileChannel sourceChannel = FileChannel.open(Paths.get(source), StandardOpenOption.READ);
             FileChannel destChannel = FileChannel.open(Paths.get(destination),
                 StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {

            ByteBuffer buffer = ByteBuffer.allocate(1024);

            while (sourceChannel.read(buffer) > 0) {
                buffer.flip(); // Switch to read mode
                destChannel.write(buffer);
                buffer.clear(); // Switch to write mode
            }
        }
    }

    private static void readLargeFileEfficiently(String filename) throws IOException {
        try (RandomAccessFile file = new RandomAccessFile(filename, "r");
             FileChannel channel = file.getChannel()) {

            // Memory-mapped file for large files
            MappedByteBuffer buffer = channel.map(
                FileChannel.MapMode.READ_ONLY, 0, channel.size());

            // TODO: Process the mapped buffer efficiently
            // This is much faster for large files
        }
    }
}
```

### 📝 Quiz Kiểm Tra Kiến Thức

**Câu 10:** Sự khác biệt chính giữa I/O và NIO?
- A) NIO nhanh hơn I/O
- B) NIO là non-blocking, I/O là blocking
- C) NIO sử dụng channels và buffers
- D) Tất cả đều đúng

**Đáp án:** 10-D

---

## 7. JVM và Garbage Collection

### 🎯 Mục tiêu học tập
- Hiểu JVM memory model
- Biết cách tune GC parameters
- Sử dụng profiling tools

### 💻 Code Mẫu và Bài Tập

#### Bài tập 11: Memory Leak Detection
```java
import java.util.*;
import java.lang.management.*;

public class MemoryLeakExample {
    private static List<byte[]> memoryLeak = new ArrayList<>();

    public static void main(String[] args) {
        // Monitor memory usage
        MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();

        Timer timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                MemoryUsage heapUsage = memoryBean.getHeapMemoryUsage();
                System.out.println("Heap Memory Used: " +
                    heapUsage.getUsed() / (1024 * 1024) + " MB");
            }
        }, 0, 1000);

        // Create memory leak
        simulateMemoryLeak();
    }

    private static void simulateMemoryLeak() {
        while (true) {
            // This creates a memory leak
            byte[] leak = new byte[1024 * 1024]; // 1MB
            memoryLeak.add(leak);

            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                break;
            }

            // TODO: Add condition to break and fix the leak
        }
    }

    // TODO: Implement method to detect and fix memory leaks
    public static void analyzeMemoryUsage() {
        // Use MemoryMXBean to get detailed memory information
        // Implement heap dump analysis
    }
}
```

### 📝 Quiz Kiểm Tra Kiến Thức

**Câu 11:** Garbage Collection nào phù hợp cho low-latency applications?
- A) Serial GC
- B) Parallel GC
- C) G1 GC
- D) ZGC

**Đáp án:** 11-D

---

## 8. Performance Optimization

### 🎯 Mục tiêu học tập
- Identify performance bottlenecks
- Optimize algorithms và data structures
- Memory và CPU optimization

### 💻 Code Mẫu và Bài Tập

#### Bài tập 12: Algorithm Optimization
```java
import java.util.concurrent.TimeUnit;

public class PerformanceOptimization {

    // Unoptimized version
    public static long fibonacciRecursive(int n) {
        if (n <= 1) return n;
        return fibonacciRecursive(n - 1) + fibonacciRecursive(n - 2);
    }

    // Optimized version with memoization
    private static Map<Integer, Long> fibCache = new HashMap<>();

    public static long fibonacciMemoized(int n) {
        if (n <= 1) return n;

        return fibCache.computeIfAbsent(n,
            k -> fibonacciMemoized(k - 1) + fibonacciMemoized(k - 2));
    }

    // Most optimized - iterative
    public static long fibonacciIterative(int n) {
        if (n <= 1) return n;

        long prev = 0, curr = 1;
        for (int i = 2; i <= n; i++) {
            long next = prev + curr;
            prev = curr;
            curr = next;
        }
        return curr;
    }

    // Benchmark method
    public static void benchmark() {
        int n = 40;

        long start = System.nanoTime();
        long result1 = fibonacciRecursive(n);
        long time1 = System.nanoTime() - start;

        start = System.nanoTime();
        long result2 = fibonacciMemoized(n);
        long time2 = System.nanoTime() - start;

        start = System.nanoTime();
        long result3 = fibonacciIterative(n);
        long time3 = System.nanoTime() - start;

        System.out.println("Recursive: " + TimeUnit.NANOSECONDS.toMillis(time1) + "ms");
        System.out.println("Memoized: " + TimeUnit.NANOSECONDS.toMillis(time2) + "ms");
        System.out.println("Iterative: " + TimeUnit.NANOSECONDS.toMillis(time3) + "ms");
    }
}
```

---

## 9. Project Thực Hành: Multi-threaded Task Scheduler

### 🎯 Mục tiêu dự án
Tạo một task scheduler với các tính năng:
- Thread-safe task queue
- Priority-based scheduling
- Monitoring và metrics
- Graceful shutdown

### 💻 Project Structure
```
task-scheduler/
├── src/main/java/
│   ├── scheduler/
│   │   ├── TaskScheduler.java
│   │   ├── Task.java
│   │   ├── TaskQueue.java
│   │   └── SchedulerMetrics.java
│   └── examples/
│       └── SchedulerDemo.java
├── src/test/java/
│   └── scheduler/
│       ├── TaskSchedulerTest.java
│       └── TaskQueueTest.java
└── pom.xml
```

### 🚀 Implementation Checklist
- [ ] Design Task interface với priority
- [ ] Implement thread-safe PriorityQueue
- [ ] Create ThreadPoolExecutor với custom ThreadFactory
- [ ] Add metrics collection (completed tasks, average execution time)
- [ ] Implement graceful shutdown mechanism
- [ ] Write comprehensive unit tests
- [ ] Add integration tests
- [ ] Performance benchmarking
- [ ] Documentation và usage examples

---

## 📚 Tài Liệu Tham Khảo Bổ Sung

### Books
1. **"Effective Java" by Joshua Bloch** - Best practices
2. **"Java Concurrency in Practice" by Brian Goetz** - Threading
3. **"Java Performance" by Scott Oaks** - Performance tuning

### Online Resources
1. **Oracle Java Documentation** - Official docs
2. **Baeldung** - Practical tutorials
3. **DZone Java Zone** - Articles và tutorials

### Tools
1. **VisualVM** - Profiling và monitoring
2. **JProfiler** - Advanced profiling
3. **Eclipse MAT** - Memory analysis
4. **JMH** - Microbenchmarking

---

## 🎯 Lời Khuyên Cuối Cùng

### Chiến lược học tập hiệu quả:
1. **Thực hành hàng ngày:** Code ít nhất 1 giờ/ngày
2. **Project-based learning:** Áp dụng kiến thức vào project thực tế
3. **Code review:** Đọc code của người khác, tham gia open source
4. **Community:** Tham gia Java communities, forums
5. **Continuous learning:** Java ecosystem luôn phát triển

### Timeline đề xuất (3 tháng):
- **Tháng 1:** Collections, Generics, Basic Threading
- **Tháng 2:** Advanced Threading, Design Patterns, Testing
- **Tháng 3:** I/O, JVM, Performance, Project

**Chúc bạn học tập hiệu quả và thành công! 🚀**
