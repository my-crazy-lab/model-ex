# Java Nâng Cao - Bộ Câu Hỏi Trắc Nghiệm

## Phần 1: Collections Framework (20 câu)

### Câu 1: ArrayList vs LinkedList
Khi nào nên sử dụng LinkedList thay vì ArrayList?
- A) Khi cần truy cập ngẫu nhiên các phần tử
- B) Khi thường xuyên thêm/xóa phần tử ở giữa danh sách
- C) Khi cần sắp xếp danh sách
- D) Khi cần thread-safe collection

**Đáp án: B**
**Giải thích:** LinkedList hiệu quả hơn cho việc thêm/xóa ở giữa danh sách (O(1) nếu có reference đến node), trong khi ArrayList cần shift các phần tử (O(n)).

### Câu 2: HashMap Internal
HashMap xử lý hash collision như thế nào trong Java 8+?
- A) Chỉ sử dụng linked list
- B) Chỉ sử dụng red-black tree
- C) Sử dụng linked list, chuyển thành tree khi bucket có >8 phần tử
- D) Tạo HashMap mới

**Đáp án: C**
**Giải thích:** Java 8 cải thiện hiệu năng bằng cách chuyển từ linked list sang red-black tree khi bucket có nhiều hơn 8 phần tử.

### Câu 3: ConcurrentHashMap
Sự khác biệt chính giữa HashMap và ConcurrentHashMap?
- A) ConcurrentHashMap nhanh hơn
- B) ConcurrentHashMap thread-safe, HashMap không
- C) ConcurrentHashMap không cho phép null values
- D) Tất cả đều đúng

**Đáp án: D**
**Giải thích:** ConcurrentHashMap thread-safe, thường nhanh hơn trong môi trường đa luồng, và không cho phép null keys/values.

### Câu 4: TreeMap vs HashMap
TreeMap có complexity nào cho get/put operations?
- A) O(1)
- B) O(log n)
- C) O(n)
- D) O(n log n)

**Đáp án: B**
**Giải thích:** TreeMap sử dụng red-black tree, có complexity O(log n) cho các operations cơ bản.

### Câu 5: Set Implementations
LinkedHashSet khác gì so với HashSet?
- A) LinkedHashSet nhanh hơn
- B) LinkedHashSet duy trì insertion order
- C) LinkedHashSet cho phép duplicate elements
- D) Không có sự khác biệt

**Đáp án: B**
**Giải thích:** LinkedHashSet duy trì thứ tự insertion bằng cách sử dụng doubly-linked list.

---

## Phần 2: Generics và Type System (15 câu)

### Câu 6: Type Erasure
Type erasure trong Java có nghĩa là gì?
- A) Generic type information bị xóa tại compile time
- B) Generic type information bị xóa tại runtime
- C) Không thể sử dụng generics với primitive types
- D) Generic chỉ hoạt động với Object

**Đáp án: B**
**Giải thích:** Type erasure xảy ra tại runtime, generic type information bị xóa để maintain backward compatibility.

### Câu 7: Bounded Types
`<T extends Comparable<T>>` có nghĩa là gì?
- A) T phải implement Comparable interface
- B) T phải extend Comparable class
- C) T phải có method compareTo
- D) A và C đều đúng

**Đáp án: D**
**Giải thích:** `extends` trong generic context có nghĩa là implement (cho interface) hoặc extend (cho class).

### Câu 8: Wildcards
Sự khác biệt giữa `List<? extends Number>` và `List<? super Number>`?
- A) Không có sự khác biệt
- B) Extends cho read, super cho write
- C) Extends cho write, super cho read
- D) Cả hai đều read-only

**Đáp án: B**
**Giải thích:** `? extends` (upper bound) cho phép read safely, `? super` (lower bound) cho phép write safely.

---

## Phần 3: Multithreading và Concurrency (25 câu)

### Câu 9: Thread States
Thread có thể chuyển trực tiếp từ WAITING sang RUNNING không?
- A) Có
- B) Không, phải qua RUNNABLE trước
- C) Chỉ khi sử dụng interrupt()
- D) Tùy thuộc vào JVM implementation

**Đáp án: B**
**Giải thích:** Thread phải qua state RUNNABLE trước khi được scheduler chọn để RUNNING.

### Câu 10: Synchronization
Sự khác biệt giữa synchronized method và synchronized block?
- A) Synchronized method nhanh hơn
- B) Synchronized block cho phép fine-grained locking
- C) Synchronized method thread-safe hơn
- D) Không có sự khác biệt

**Đáp án: B**
**Giải thích:** Synchronized block cho phép lock chỉ một phần code và có thể sử dụng different lock objects.

### Câu 11: volatile keyword
volatile keyword đảm bảo điều gì?
- A) Thread safety
- B) Atomicity
- C) Visibility và ordering
- D) Tất cả đều đúng

**Đáp án: C**
**Giải thích:** volatile đảm bảo visibility (changes visible to all threads) và prevents reordering, nhưng không đảm bảo atomicity.

### Câu 12: AtomicInteger
AtomicInteger.incrementAndGet() có thread-safe không?
- A) Có, sử dụng CAS operations
- B) Không, cần thêm synchronization
- C) Chỉ thread-safe trong single-core systems
- D) Tùy thuộc vào JVM

**Đáp án: A**
**Giải thích:** AtomicInteger sử dụng Compare-And-Swap (CAS) operations, là lock-free và thread-safe.

### Câu 13: Deadlock
Điều kiện nào KHÔNG cần thiết để xảy ra deadlock?
- A) Mutual exclusion
- B) Hold and wait
- C) No preemption
- D) High CPU usage

**Đáp án: D**
**Giải thích:** 4 điều kiện cần thiết cho deadlock: mutual exclusion, hold and wait, no preemption, circular wait. CPU usage không liên quan.

### Câu 14: ExecutorService
Sự khác biệt giữa submit() và execute() trong ExecutorService?
- A) submit() trả về Future, execute() không
- B) execute() nhanh hơn submit()
- C) submit() chỉ dùng cho Callable
- D) Không có sự khác biệt

**Đáp án: A**
**Giải thích:** submit() trả về Future object để track task completion và get result, execute() không trả về gì.

### Câu 15: CountDownLatch
CountDownLatch có thể reset và reuse được không?
- A) Có, sử dụng reset() method
- B) Không, chỉ sử dụng một lần
- C) Có, nhưng cần synchronization
- D) Tùy thuộc vào initial count

**Đáp án: B**
**Giải thích:** CountDownLatch không thể reset. Để reuse, cần sử dụng CyclicBarrier.

---

## Phần 4: Design Patterns (15 câu)

### Câu 16: Singleton Pattern
Cách implement Singleton thread-safe và lazy initialization?
- A) Synchronized method
- B) Double-checked locking
- C) Enum singleton
- D) Tất cả đều đúng

**Đáp án: D**
**Giải thích:** Tất cả đều là cách implement Singleton thread-safe, mỗi cách có ưu nhược điểm riêng.

### Câu 17: Observer Pattern
Observer pattern giải quyết vấn đề gì?
- A) Tạo objects
- B) Loose coupling giữa subject và observers
- C) Thread safety
- D) Memory management

**Đáp án: B**
**Giải thích:** Observer pattern cho phép loose coupling, subject không cần biết chi tiết về observers.

### Câu 18: Strategy Pattern
Khi nào nên sử dụng Strategy pattern?
- A) Khi có nhiều cách implement một algorithm
- B) Khi muốn thay đổi algorithm tại runtime
- C) Khi muốn tránh if-else chains
- D) Tất cả đều đúng

**Đáp án: D**
**Giải thích:** Strategy pattern hữu ích trong tất cả các trường hợp trên.

---

## Phần 5: JVM và Performance (10 câu)

### Câu 19: Garbage Collection
G1 GC phù hợp cho application nào?
- A) Small heap applications
- B) Large heap applications với low latency requirements
- C) CPU-intensive applications
- D) Single-threaded applications

**Đáp án: B**
**Giải thích:** G1 GC được thiết kế cho large heap (>4GB) với predictable pause times.

### Câu 20: Memory Areas
Method area trong JVM chứa gì?
- A) Instance variables
- B) Local variables
- C) Class metadata và constant pool
- D) Thread stacks

**Đáp án: C**
**Giải thích:** Method area (Metaspace trong Java 8+) chứa class metadata, constant pool, static variables.

---

## Phần 6: I/O và NIO (10 câu)

### Câu 21: NIO vs IO
Sự khác biệt chính giữa NIO và traditional I/O?
- A) NIO nhanh hơn
- B) NIO non-blocking, IO blocking
- C) NIO sử dụng channels và buffers
- D) B và C đều đúng

**Đáp án: D**
**Giải thích:** NIO cung cấp non-blocking I/O và sử dụng channels/buffers thay vì streams.

### Câu 22: Files API
Files.walk() trả về gì?
- A) List<Path>
- B) Stream<Path>
- C) Iterator<Path>
- D) Array of Paths

**Đáp án: B**
**Giải thích:** Files.walk() trả về Stream<Path> để lazy evaluation và memory efficiency.

---

## Phần 7: Testing và Best Practices (5 câu)

### Câu 23: JUnit 5
@BeforeEach và @BeforeAll khác nhau như thế nào?
- A) @BeforeEach chạy trước mỗi test method
- B) @BeforeAll chạy một lần trước tất cả tests
- C) @BeforeAll method phải static
- D) Tất cả đều đúng

**Đáp án: D**
**Giải thích:** @BeforeEach chạy trước mỗi test, @BeforeAll chạy một lần và phải static.

### Câu 24: Mockito
verify() trong Mockito dùng để làm gì?
- A) Verify method được gọi
- B) Verify số lần method được gọi
- C) Verify parameters của method call
- D) Tất cả đều đúng

**Đáp án: D**
**Giải thích:** verify() có thể check method calls, số lần gọi, và parameters.

### Câu 25: TDD
TDD workflow đúng là gì?
- A) Code -> Test -> Refactor
- B) Test -> Code -> Refactor (Red-Green-Refactor)
- C) Design -> Code -> Test
- D) Refactor -> Test -> Code

**Đáp án: B**
**Giải thích:** TDD follow Red-Green-Refactor cycle: viết failing test, implement code để pass, refactor.

---

## Scoring Guide

- **23-25 điểm:** Excellent - Bạn đã master Java nâng cao
- **20-22 điểm:** Good - Cần ôn lại một số concepts
- **17-19 điểm:** Average - Cần học thêm và thực hành nhiều hơn
- **< 17 điểm:** Needs Improvement - Nên focus vào fundamentals trước

## Đáp Án Tổng Hợp
1.B 2.C 3.D 4.B 5.B 6.B 7.D 8.B 9.B 10.B 11.C 12.A 13.D 14.A 15.B 16.D 17.B 18.D 19.B 20.C 21.D 22.B 23.D 24.D 25.B
