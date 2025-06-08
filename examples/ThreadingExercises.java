package examples;

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Bài tập thực hành Multithreading và Concurrency
 * 
 * TODO cho học viên:
 * 1. Complete các method có TODO
 * 2. Run và observe thread behavior
 * 3. Fix race conditions và deadlocks
 * 4. Implement additional synchronization mechanisms
 */
public class ThreadingExercises {
    
    public static void main(String[] args) throws InterruptedException {
        System.out.println("=== Threading Exercises ===\n");
        
        // Exercise 1: Basic thread creation
        demonstrateBasicThreads();
        
        // Exercise 2: Producer-Consumer
        demonstrateProducerConsumer();
        
        // Exercise 3: Race condition
        demonstrateRaceCondition();
        
        // Exercise 4: Deadlock
        demonstrateDeadlock();
        
        // Exercise 5: Thread pool
        demonstrateThreadPool();
    }
    
    /**
     * Exercise 1: Basic Thread Creation
     */
    public static void demonstrateBasicThreads() throws InterruptedException {
        System.out.println("--- Basic Threads ---");
        
        // Method 1: Extending Thread class
        Thread thread1 = new Thread() {
            @Override
            public void run() {
                for (int i = 0; i < 5; i++) {
                    System.out.println("Thread 1: " + i);
                    try {
                        Thread.sleep(100);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }
        };
        
        // Method 2: Implementing Runnable
        Runnable task = () -> {
            for (int i = 0; i < 5; i++) {
                System.out.println("Thread 2: " + i);
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        };
        Thread thread2 = new Thread(task);
        
        thread1.start();
        thread2.start();
        
        thread1.join();
        thread2.join();
        
        System.out.println("Basic threads completed\n");
    }
    
    /**
     * Exercise 2: Producer-Consumer Pattern
     */
    public static void demonstrateProducerConsumer() throws InterruptedException {
        System.out.println("--- Producer-Consumer ---");
        
        BlockingQueue<String> queue = new ArrayBlockingQueue<>(5);
        AtomicInteger itemCount = new AtomicInteger(0);
        
        // Producer
        Thread producer = new Thread(() -> {
            try {
                for (int i = 0; i < 10; i++) {
                    String item = "Item-" + itemCount.incrementAndGet();
                    queue.put(item);
                    System.out.println("Produced: " + item + " (Queue size: " + queue.size() + ")");
                    Thread.sleep(100);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });
        
        // Consumer
        Thread consumer = new Thread(() -> {
            try {
                while (!Thread.currentThread().isInterrupted()) {
                    String item = queue.poll(200, TimeUnit.MILLISECONDS);
                    if (item != null) {
                        System.out.println("Consumed: " + item + " (Queue size: " + queue.size() + ")");
                        Thread.sleep(150);
                    } else {
                        break; // Timeout, assume no more items
                    }
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });
        
        producer.start();
        consumer.start();
        
        producer.join();
        consumer.join();
        
        System.out.println("Producer-Consumer completed\n");
    }
    
    /**
     * Exercise 3: Race Condition Demo and Fix
     */
    public static void demonstrateRaceCondition() throws InterruptedException {
        System.out.println("--- Race Condition ---");
        
        // Unsafe counter
        UnsafeCounter unsafeCounter = new UnsafeCounter();
        
        // Safe counter
        SafeCounter safeCounter = new SafeCounter();
        
        int numThreads = 10;
        int incrementsPerThread = 1000;
        
        // Test unsafe counter
        Thread[] unsafeThreads = new Thread[numThreads];
        for (int i = 0; i < numThreads; i++) {
            unsafeThreads[i] = new Thread(() -> {
                for (int j = 0; j < incrementsPerThread; j++) {
                    unsafeCounter.increment();
                }
            });
        }
        
        long start = System.currentTimeMillis();
        for (Thread t : unsafeThreads) t.start();
        for (Thread t : unsafeThreads) t.join();
        long unsafeTime = System.currentTimeMillis() - start;
        
        // Test safe counter
        Thread[] safeThreads = new Thread[numThreads];
        for (int i = 0; i < numThreads; i++) {
            safeThreads[i] = new Thread(() -> {
                for (int j = 0; j < incrementsPerThread; j++) {
                    safeCounter.increment();
                }
            });
        }
        
        start = System.currentTimeMillis();
        for (Thread t : safeThreads) t.start();
        for (Thread t : safeThreads) t.join();
        long safeTime = System.currentTimeMillis() - start;
        
        int expected = numThreads * incrementsPerThread;
        System.out.println("Expected: " + expected);
        System.out.println("Unsafe counter: " + unsafeCounter.getValue() + " (Time: " + unsafeTime + "ms)");
        System.out.println("Safe counter: " + safeCounter.getValue() + " (Time: " + safeTime + "ms)");
        System.out.println();
    }
    
    /**
     * Exercise 4: Deadlock Demo
     * TODO: Implement deadlock scenario and prevention
     */
    public static void demonstrateDeadlock() {
        System.out.println("--- Deadlock Demo ---");
        
        Object lock1 = new Object();
        Object lock2 = new Object();
        
        Thread thread1 = new Thread(() -> {
            synchronized (lock1) {
                System.out.println("Thread 1: Holding lock 1...");
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                
                System.out.println("Thread 1: Waiting for lock 2...");
                synchronized (lock2) {
                    System.out.println("Thread 1: Holding lock 1 & 2");
                }
            }
        });
        
        Thread thread2 = new Thread(() -> {
            synchronized (lock2) {
                System.out.println("Thread 2: Holding lock 2...");
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                
                System.out.println("Thread 2: Waiting for lock 1...");
                synchronized (lock1) {
                    System.out.println("Thread 2: Holding lock 1 & 2");
                }
            }
        });
        
        thread1.start();
        thread2.start();
        
        // TODO: Add deadlock detection and prevention
        // Hint: Use timeout, ordered locking, or tryLock()
        
        try {
            thread1.join(2000); // Wait max 2 seconds
            thread2.join(2000);
            
            if (thread1.isAlive() || thread2.isAlive()) {
                System.out.println("Deadlock detected! Interrupting threads...");
                thread1.interrupt();
                thread2.interrupt();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        System.out.println("Deadlock demo completed\n");
    }
    
    /**
     * Exercise 5: Thread Pool Usage
     */
    public static void demonstrateThreadPool() throws InterruptedException {
        System.out.println("--- Thread Pool ---");
        
        // Fixed thread pool
        ExecutorService fixedPool = Executors.newFixedThreadPool(3);
        
        // Submit tasks
        for (int i = 0; i < 10; i++) {
            final int taskId = i;
            fixedPool.submit(() -> {
                System.out.println("Task " + taskId + " executed by " + Thread.currentThread().getName());
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
        }
        
        // Graceful shutdown
        fixedPool.shutdown();
        if (!fixedPool.awaitTermination(10, TimeUnit.SECONDS)) {
            fixedPool.shutdownNow();
        }
        
        System.out.println("Thread pool completed\n");
    }
    
    // Helper classes
    static class UnsafeCounter {
        private int count = 0;
        
        public void increment() {
            count++; // Not thread-safe!
        }
        
        public int getValue() {
            return count;
        }
    }
    
    static class SafeCounter {
        private final AtomicInteger count = new AtomicInteger(0);
        
        public void increment() {
            count.incrementAndGet(); // Thread-safe
        }
        
        public int getValue() {
            return count.get();
        }
    }
    
    // TODO: Implement additional exercises
    // 1. ReadWriteLock example
    // 2. CountDownLatch usage
    // 3. Semaphore for resource limiting
    // 4. CompletableFuture for async programming
}
