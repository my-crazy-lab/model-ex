package examples;

import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Design Patterns Implementation Examples
 * 
 * TODO cho học viên:
 * 1. Complete các TODO methods
 * 2. Implement additional patterns (Factory, Decorator, Command)
 * 3. Create real-world examples for each pattern
 * 4. Write unit tests for each pattern
 */
public class DesignPatternsDemo {
    
    public static void main(String[] args) {
        System.out.println("=== Design Patterns Demo ===\n");
        
        // Singleton Pattern
        demonstrateSingleton();
        
        // Observer Pattern
        demonstrateObserver();
        
        // Strategy Pattern
        demonstrateStrategy();
        
        // Factory Pattern
        demonstrateFactory();
        
        // Builder Pattern
        demonstrateBuilder();
    }
    
    /**
     * Singleton Pattern Demo
     */
    public static void demonstrateSingleton() {
        System.out.println("--- Singleton Pattern ---");
        
        DatabaseConnection db1 = DatabaseConnection.getInstance();
        DatabaseConnection db2 = DatabaseConnection.getInstance();
        
        System.out.println("Same instance: " + (db1 == db2));
        
        db1.connect();
        db2.executeQuery("SELECT * FROM users");
        
        System.out.println();
    }
    
    /**
     * Observer Pattern Demo
     */
    public static void demonstrateObserver() {
        System.out.println("--- Observer Pattern ---");
        
        NewsAgency agency = new NewsAgency();
        
        NewsChannel cnn = new NewsChannel("CNN");
        NewsChannel bbc = new NewsChannel("BBC");
        
        agency.addObserver(cnn);
        agency.addObserver(bbc);
        
        agency.setNews("Breaking: Java 21 Released!");
        agency.setNews("Tech: New AI Breakthrough");
        
        agency.removeObserver(bbc);
        agency.setNews("Sports: World Cup Finals");
        
        System.out.println();
    }
    
    /**
     * Strategy Pattern Demo
     */
    public static void demonstrateStrategy() {
        System.out.println("--- Strategy Pattern ---");
        
        ShoppingCart cart = new ShoppingCart();
        cart.addItem("Laptop", 1000.0);
        cart.addItem("Mouse", 25.0);
        
        // Pay with credit card
        cart.setPaymentStrategy(new CreditCardPayment("1234-5678-9012-3456"));
        cart.checkout();
        
        // Pay with PayPal
        cart.setPaymentStrategy(new PayPalPayment("user@example.com"));
        cart.checkout();
        
        // TODO: Implement Bitcoin payment strategy
        
        System.out.println();
    }
    
    /**
     * Factory Pattern Demo
     */
    public static void demonstrateFactory() {
        System.out.println("--- Factory Pattern ---");
        
        // Simple Factory
        Animal dog = AnimalFactory.createAnimal("DOG");
        Animal cat = AnimalFactory.createAnimal("CAT");
        
        dog.makeSound();
        cat.makeSound();
        
        // TODO: Implement Abstract Factory for different animal families
        
        System.out.println();
    }
    
    /**
     * Builder Pattern Demo
     */
    public static void demonstrateBuilder() {
        System.out.println("--- Builder Pattern ---");
        
        Computer computer = new Computer.Builder()
            .setCPU("Intel i7")
            .setRAM("16GB")
            .setStorage("512GB SSD")
            .setGPU("NVIDIA RTX 3080")
            .build();
        
        System.out.println(computer);
        
        // Minimal configuration
        Computer basicComputer = new Computer.Builder()
            .setCPU("Intel i3")
            .setRAM("8GB")
            .build();
        
        System.out.println(basicComputer);
        
        System.out.println();
    }
}

// ============ SINGLETON PATTERN ============
class DatabaseConnection {
    private static volatile DatabaseConnection instance;
    private boolean connected = false;
    
    private DatabaseConnection() {
        // Private constructor
    }
    
    public static DatabaseConnection getInstance() {
        if (instance == null) {
            synchronized (DatabaseConnection.class) {
                if (instance == null) {
                    instance = new DatabaseConnection();
                }
            }
        }
        return instance;
    }
    
    public void connect() {
        if (!connected) {
            System.out.println("Database connected");
            connected = true;
        }
    }
    
    public void executeQuery(String query) {
        if (connected) {
            System.out.println("Executing: " + query);
        } else {
            System.out.println("Not connected to database");
        }
    }
}

// ============ OBSERVER PATTERN ============
interface Observer {
    void update(String news);
}

class NewsAgency {
    private List<Observer> observers = new CopyOnWriteArrayList<>();
    private String news;
    
    public void addObserver(Observer observer) {
        observers.add(observer);
    }
    
    public void removeObserver(Observer observer) {
        observers.remove(observer);
    }
    
    public void setNews(String news) {
        this.news = news;
        notifyObservers();
    }
    
    private void notifyObservers() {
        for (Observer observer : observers) {
            observer.update(news);
        }
    }
}

class NewsChannel implements Observer {
    private String name;
    
    public NewsChannel(String name) {
        this.name = name;
    }
    
    @Override
    public void update(String news) {
        System.out.println(name + " received news: " + news);
    }
}

// ============ STRATEGY PATTERN ============
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
        System.out.println("Paid $" + amount + " using Credit Card ending in " + 
                          cardNumber.substring(cardNumber.length() - 4));
    }
}

class PayPalPayment implements PaymentStrategy {
    private String email;
    
    public PayPalPayment(String email) {
        this.email = email;
    }
    
    @Override
    public void pay(double amount) {
        System.out.println("Paid $" + amount + " using PayPal account: " + email);
    }
}

class ShoppingCart {
    private List<Item> items = new ArrayList<>();
    private PaymentStrategy paymentStrategy;
    
    public void addItem(String name, double price) {
        items.add(new Item(name, price));
    }
    
    public void setPaymentStrategy(PaymentStrategy strategy) {
        this.paymentStrategy = strategy;
    }
    
    public void checkout() {
        double total = items.stream().mapToDouble(Item::getPrice).sum();
        System.out.println("Total amount: $" + total);
        
        if (paymentStrategy != null) {
            paymentStrategy.pay(total);
        } else {
            System.out.println("No payment method selected");
        }
    }
    
    static class Item {
        private String name;
        private double price;
        
        public Item(String name, double price) {
            this.name = name;
            this.price = price;
        }
        
        public double getPrice() { return price; }
        public String getName() { return name; }
    }
}

// ============ FACTORY PATTERN ============
interface Animal {
    void makeSound();
}

class Dog implements Animal {
    @Override
    public void makeSound() {
        System.out.println("Woof! Woof!");
    }
}

class Cat implements Animal {
    @Override
    public void makeSound() {
        System.out.println("Meow! Meow!");
    }
}

class AnimalFactory {
    public static Animal createAnimal(String type) {
        switch (type.toUpperCase()) {
            case "DOG":
                return new Dog();
            case "CAT":
                return new Cat();
            default:
                throw new IllegalArgumentException("Unknown animal type: " + type);
        }
    }
}

// ============ BUILDER PATTERN ============
class Computer {
    private String CPU;
    private String RAM;
    private String storage;
    private String GPU;
    private boolean hasWiFi;
    
    private Computer(Builder builder) {
        this.CPU = builder.CPU;
        this.RAM = builder.RAM;
        this.storage = builder.storage;
        this.GPU = builder.GPU;
        this.hasWiFi = builder.hasWiFi;
    }
    
    @Override
    public String toString() {
        return "Computer{" +
                "CPU='" + CPU + '\'' +
                ", RAM='" + RAM + '\'' +
                ", storage='" + storage + '\'' +
                ", GPU='" + GPU + '\'' +
                ", hasWiFi=" + hasWiFi +
                '}';
    }
    
    static class Builder {
        private String CPU;
        private String RAM;
        private String storage;
        private String GPU;
        private boolean hasWiFi = true; // Default value
        
        public Builder setCPU(String CPU) {
            this.CPU = CPU;
            return this;
        }
        
        public Builder setRAM(String RAM) {
            this.RAM = RAM;
            return this;
        }
        
        public Builder setStorage(String storage) {
            this.storage = storage;
            return this;
        }
        
        public Builder setGPU(String GPU) {
            this.GPU = GPU;
            return this;
        }
        
        public Builder setWiFi(boolean hasWiFi) {
            this.hasWiFi = hasWiFi;
            return this;
        }
        
        public Computer build() {
            // Validation
            if (CPU == null || RAM == null) {
                throw new IllegalStateException("CPU and RAM are required");
            }
            return new Computer(this);
        }
    }
}
