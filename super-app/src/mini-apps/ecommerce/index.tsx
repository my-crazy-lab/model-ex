/**
 * E-commerce Mini-App
 * 
 * A complete e-commerce experience within the Super App
 */

import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  FlatList,
  Alert,
  RefreshControl,
} from 'react-native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { NavigationContainer } from '@react-navigation/native';
import Icon from 'react-native-vector-icons/MaterialIcons';

// Shared components and services
import { LoadingScreen } from '@shared/components/LoadingScreen';
import { ErrorBoundary } from '@shared/components/ErrorBoundary';
import { AnalyticsService } from '@shared/services/AnalyticsService';

// Types
interface Product {
  id: string;
  name: string;
  price: number;
  image: string;
  description: string;
  category: string;
  rating: number;
  reviews: number;
  inStock: boolean;
}

interface CartItem extends Product {
  quantity: number;
}

// Navigation types
type EcommerceStackParamList = {
  ProductCatalog: undefined;
  ProductDetails: { productId: string };
  ShoppingCart: undefined;
  Checkout: undefined;
  OrderConfirmation: { orderId: string };
};

const Stack = createNativeStackNavigator<EcommerceStackParamList>();

/**
 * Product Catalog Screen
 */
const ProductCatalogScreen: React.FC<{ navigation: any }> = ({ navigation }) => {
  const [products, setProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [cartItemCount, setCartItemCount] = useState(0);

  useEffect(() => {
    loadProducts();
    
    // Track screen view
    AnalyticsService.trackScreenView('ecommerce_product_catalog');
  }, []);

  const loadProducts = async () => {
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const mockProducts: Product[] = [
        {
          id: '1',
          name: 'Wireless Headphones',
          price: 99.99,
          image: 'https://via.placeholder.com/200x200/007AFF/FFFFFF?text=Headphones',
          description: 'High-quality wireless headphones with noise cancellation',
          category: 'Electronics',
          rating: 4.5,
          reviews: 128,
          inStock: true,
        },
        {
          id: '2',
          name: 'Smart Watch',
          price: 199.99,
          image: 'https://via.placeholder.com/200x200/34C759/FFFFFF?text=Watch',
          description: 'Feature-rich smartwatch with health monitoring',
          category: 'Electronics',
          rating: 4.3,
          reviews: 89,
          inStock: true,
        },
        {
          id: '3',
          name: 'Laptop Backpack',
          price: 49.99,
          image: 'https://via.placeholder.com/200x200/FF9500/FFFFFF?text=Backpack',
          description: 'Durable laptop backpack with multiple compartments',
          category: 'Accessories',
          rating: 4.7,
          reviews: 203,
          inStock: false,
        },
        {
          id: '4',
          name: 'Bluetooth Speaker',
          price: 79.99,
          image: 'https://via.placeholder.com/200x200/FF3B30/FFFFFF?text=Speaker',
          description: 'Portable Bluetooth speaker with excellent sound quality',
          category: 'Electronics',
          rating: 4.4,
          reviews: 156,
          inStock: true,
        },
      ];

      setProducts(mockProducts);
      setLoading(false);
      setRefreshing(false);

      // Track products loaded
      AnalyticsService.trackEvent('ecommerce_products_loaded', {
        productCount: mockProducts.length,
      });

    } catch (error) {
      console.error('Failed to load products:', error);
      setLoading(false);
      setRefreshing(false);
      
      Alert.alert('Error', 'Failed to load products. Please try again.');
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    loadProducts();
  };

  const renderProduct = ({ item }: { item: Product }) => (
    <TouchableOpacity
      style={styles.productCard}
      onPress={() => {
        navigation.navigate('ProductDetails', { productId: item.id });
        
        // Track product view
        AnalyticsService.trackEvent('ecommerce_product_viewed', {
          productId: item.id,
          productName: item.name,
          category: item.category,
        });
      }}
    >
      <Image source={{ uri: item.image }} style={styles.productImage} />
      <View style={styles.productInfo}>
        <Text style={styles.productName} numberOfLines={2}>
          {item.name}
        </Text>
        <Text style={styles.productPrice}>${item.price}</Text>
        <View style={styles.productRating}>
          <Icon name="star" size={16} color="#FFD700" />
          <Text style={styles.ratingText}>
            {item.rating} ({item.reviews})
          </Text>
        </View>
        {!item.inStock && (
          <Text style={styles.outOfStock}>Out of Stock</Text>
        )}
      </View>
    </TouchableOpacity>
  );

  if (loading) {
    return <LoadingScreen message="Loading products..." />;
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Products</Text>
        <TouchableOpacity
          style={styles.cartButton}
          onPress={() => navigation.navigate('ShoppingCart')}
        >
          <Icon name="shopping-cart" size={24} color="#007AFF" />
          {cartItemCount > 0 && (
            <View style={styles.cartBadge}>
              <Text style={styles.cartBadgeText}>{cartItemCount}</Text>
            </View>
          )}
        </TouchableOpacity>
      </View>

      <FlatList
        data={products}
        renderItem={renderProduct}
        keyExtractor={(item) => item.id}
        numColumns={2}
        contentContainerStyle={styles.productList}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      />
    </View>
  );
};

/**
 * Product Details Screen
 */
const ProductDetailsScreen: React.FC<{ route: any; navigation: any }> = ({
  route,
  navigation,
}) => {
  const { productId } = route.params;
  const [product, setProduct] = useState<Product | null>(null);
  const [loading, setLoading] = useState(true);
  const [quantity, setQuantity] = useState(1);

  useEffect(() => {
    loadProductDetails();
    
    // Track screen view
    AnalyticsService.trackScreenView('ecommerce_product_details', {
      productId,
    });
  }, [productId]);

  const loadProductDetails = async () => {
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Mock product details (in real app, fetch from API)
      const mockProduct: Product = {
        id: productId,
        name: 'Wireless Headphones',
        price: 99.99,
        image: 'https://via.placeholder.com/400x400/007AFF/FFFFFF?text=Headphones',
        description: 'High-quality wireless headphones with active noise cancellation, 30-hour battery life, and premium sound quality. Perfect for music lovers and professionals.',
        category: 'Electronics',
        rating: 4.5,
        reviews: 128,
        inStock: true,
      };

      setProduct(mockProduct);
      setLoading(false);

    } catch (error) {
      console.error('Failed to load product details:', error);
      setLoading(false);
      Alert.alert('Error', 'Failed to load product details.');
    }
  };

  const addToCart = () => {
    if (!product) return;

    // Track add to cart
    AnalyticsService.trackEvent('ecommerce_add_to_cart', {
      productId: product.id,
      productName: product.name,
      price: product.price,
      quantity,
    });

    Alert.alert(
      'Added to Cart',
      `${product.name} (${quantity}) has been added to your cart.`,
      [
        { text: 'Continue Shopping', style: 'cancel' },
        { text: 'View Cart', onPress: () => navigation.navigate('ShoppingCart') },
      ]
    );
  };

  if (loading) {
    return <LoadingScreen message="Loading product details..." />;
  }

  if (!product) {
    return (
      <View style={styles.errorContainer}>
        <Text style={styles.errorText}>Product not found</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <Image source={{ uri: product.image }} style={styles.detailImage} />
      
      <View style={styles.detailContent}>
        <Text style={styles.detailName}>{product.name}</Text>
        <Text style={styles.detailPrice}>${product.price}</Text>
        
        <View style={styles.detailRating}>
          <Icon name="star" size={20} color="#FFD700" />
          <Text style={styles.detailRatingText}>
            {product.rating} ({product.reviews} reviews)
          </Text>
        </View>

        <Text style={styles.detailDescription}>{product.description}</Text>

        <View style={styles.quantityContainer}>
          <Text style={styles.quantityLabel}>Quantity:</Text>
          <View style={styles.quantityControls}>
            <TouchableOpacity
              style={styles.quantityButton}
              onPress={() => setQuantity(Math.max(1, quantity - 1))}
            >
              <Icon name="remove" size={20} color="#007AFF" />
            </TouchableOpacity>
            <Text style={styles.quantityText}>{quantity}</Text>
            <TouchableOpacity
              style={styles.quantityButton}
              onPress={() => setQuantity(quantity + 1)}
            >
              <Icon name="add" size={20} color="#007AFF" />
            </TouchableOpacity>
          </View>
        </View>

        <TouchableOpacity
          style={[
            styles.addToCartButton,
            !product.inStock && styles.disabledButton,
          ]}
          onPress={addToCart}
          disabled={!product.inStock}
        >
          <Text style={styles.addToCartText}>
            {product.inStock ? 'Add to Cart' : 'Out of Stock'}
          </Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
};

/**
 * Shopping Cart Screen
 */
const ShoppingCartScreen: React.FC<{ navigation: any }> = ({ navigation }) => {
  const [cartItems, setCartItems] = useState<CartItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadCartItems();
    
    // Track screen view
    AnalyticsService.trackScreenView('ecommerce_shopping_cart');
  }, []);

  const loadCartItems = async () => {
    try {
      // Simulate loading cart items
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Mock cart items
      const mockCartItems: CartItem[] = [
        {
          id: '1',
          name: 'Wireless Headphones',
          price: 99.99,
          image: 'https://via.placeholder.com/100x100/007AFF/FFFFFF?text=Headphones',
          description: 'High-quality wireless headphones',
          category: 'Electronics',
          rating: 4.5,
          reviews: 128,
          inStock: true,
          quantity: 1,
        },
      ];

      setCartItems(mockCartItems);
      setLoading(false);

    } catch (error) {
      console.error('Failed to load cart items:', error);
      setLoading(false);
    }
  };

  const getTotalPrice = () => {
    return cartItems.reduce((total, item) => total + (item.price * item.quantity), 0);
  };

  const proceedToCheckout = () => {
    if (cartItems.length === 0) {
      Alert.alert('Empty Cart', 'Please add items to your cart before checkout.');
      return;
    }

    // Track checkout start
    AnalyticsService.trackEvent('ecommerce_checkout_started', {
      itemCount: cartItems.length,
      totalValue: getTotalPrice(),
    });

    navigation.navigate('Checkout');
  };

  if (loading) {
    return <LoadingScreen message="Loading cart..." />;
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Shopping Cart</Text>
      </View>

      {cartItems.length === 0 ? (
        <View style={styles.emptyCart}>
          <Icon name="shopping-cart" size={64} color="#CCCCCC" />
          <Text style={styles.emptyCartText}>Your cart is empty</Text>
          <TouchableOpacity
            style={styles.continueShoppingButton}
            onPress={() => navigation.navigate('ProductCatalog')}
          >
            <Text style={styles.continueShoppingText}>Continue Shopping</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <>
          <FlatList
            data={cartItems}
            keyExtractor={(item) => item.id}
            renderItem={({ item }) => (
              <View style={styles.cartItem}>
                <Image source={{ uri: item.image }} style={styles.cartItemImage} />
                <View style={styles.cartItemInfo}>
                  <Text style={styles.cartItemName}>{item.name}</Text>
                  <Text style={styles.cartItemPrice}>${item.price}</Text>
                  <Text style={styles.cartItemQuantity}>Qty: {item.quantity}</Text>
                </View>
              </View>
            )}
            style={styles.cartList}
          />

          <View style={styles.cartSummary}>
            <View style={styles.totalRow}>
              <Text style={styles.totalLabel}>Total:</Text>
              <Text style={styles.totalPrice}>${getTotalPrice().toFixed(2)}</Text>
            </View>
            <TouchableOpacity
              style={styles.checkoutButton}
              onPress={proceedToCheckout}
            >
              <Text style={styles.checkoutButtonText}>Proceed to Checkout</Text>
            </TouchableOpacity>
          </View>
        </>
      )}
    </View>
  );
};

/**
 * Main E-commerce Mini-App Component
 */
const EcommerceMiniApp: React.FC = () => {
  useEffect(() => {
    // Track mini-app launch
    AnalyticsService.trackEvent('miniapp_launched', {
      miniAppId: 'ecommerce',
      timestamp: new Date().toISOString(),
    });

    return () => {
      // Cleanup when mini-app is unmounted
      AnalyticsService.trackEvent('miniapp_closed', {
        miniAppId: 'ecommerce',
        timestamp: new Date().toISOString(),
      });
    };
  }, []);

  return (
    <ErrorBoundary>
      <NavigationContainer independent={true}>
        <Stack.Navigator
          screenOptions={{
            headerStyle: {
              backgroundColor: '#007AFF',
            },
            headerTintColor: '#FFFFFF',
            headerTitleStyle: {
              fontWeight: 'bold',
            },
          }}
        >
          <Stack.Screen
            name="ProductCatalog"
            component={ProductCatalogScreen}
            options={{ title: 'E-commerce' }}
          />
          <Stack.Screen
            name="ProductDetails"
            component={ProductDetailsScreen}
            options={{ title: 'Product Details' }}
          />
          <Stack.Screen
            name="ShoppingCart"
            component={ShoppingCartScreen}
            options={{ title: 'Shopping Cart' }}
          />
        </Stack.Navigator>
      </NavigationContainer>
    </ErrorBoundary>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E5EA',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#000000',
  },
  cartButton: {
    position: 'relative',
    padding: 8,
  },
  cartBadge: {
    position: 'absolute',
    top: 0,
    right: 0,
    backgroundColor: '#FF3B30',
    borderRadius: 10,
    minWidth: 20,
    height: 20,
    justifyContent: 'center',
    alignItems: 'center',
  },
  cartBadgeText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: 'bold',
  },
  productList: {
    padding: 16,
  },
  productCard: {
    flex: 1,
    margin: 8,
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  productImage: {
    width: '100%',
    height: 150,
    borderTopLeftRadius: 12,
    borderTopRightRadius: 12,
  },
  productInfo: {
    padding: 12,
  },
  productName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#000000',
    marginBottom: 4,
  },
  productPrice: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#007AFF',
    marginBottom: 4,
  },
  productRating: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  ratingText: {
    marginLeft: 4,
    fontSize: 14,
    color: '#666666',
  },
  outOfStock: {
    marginTop: 4,
    fontSize: 12,
    color: '#FF3B30',
    fontWeight: '500',
  },
  detailImage: {
    width: '100%',
    height: 300,
  },
  detailContent: {
    padding: 16,
  },
  detailName: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#000000',
    marginBottom: 8,
  },
  detailPrice: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#007AFF',
    marginBottom: 8,
  },
  detailRating: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  detailRatingText: {
    marginLeft: 8,
    fontSize: 16,
    color: '#666666',
  },
  detailDescription: {
    fontSize: 16,
    lineHeight: 24,
    color: '#333333',
    marginBottom: 24,
  },
  quantityContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 24,
  },
  quantityLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#000000',
    marginRight: 16,
  },
  quantityControls: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#E5E5EA',
    borderRadius: 8,
  },
  quantityButton: {
    padding: 12,
  },
  quantityText: {
    paddingHorizontal: 16,
    fontSize: 16,
    fontWeight: '600',
    color: '#000000',
  },
  addToCartButton: {
    backgroundColor: '#007AFF',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
  },
  disabledButton: {
    backgroundColor: '#CCCCCC',
  },
  addToCartText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: 'bold',
  },
  emptyCart: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 32,
  },
  emptyCartText: {
    fontSize: 18,
    color: '#666666',
    marginTop: 16,
    marginBottom: 24,
  },
  continueShoppingButton: {
    backgroundColor: '#007AFF',
    borderRadius: 12,
    paddingHorizontal: 24,
    paddingVertical: 12,
  },
  continueShoppingText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  cartList: {
    flex: 1,
  },
  cartItem: {
    flexDirection: 'row',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E5EA',
  },
  cartItemImage: {
    width: 80,
    height: 80,
    borderRadius: 8,
  },
  cartItemInfo: {
    flex: 1,
    marginLeft: 16,
    justifyContent: 'center',
  },
  cartItemName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#000000',
    marginBottom: 4,
  },
  cartItemPrice: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#007AFF',
    marginBottom: 4,
  },
  cartItemQuantity: {
    fontSize: 14,
    color: '#666666',
  },
  cartSummary: {
    padding: 16,
    borderTopWidth: 1,
    borderTopColor: '#E5E5EA',
    backgroundColor: '#F8F9FA',
  },
  totalRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  totalLabel: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#000000',
  },
  totalPrice: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#007AFF',
  },
  checkoutButton: {
    backgroundColor: '#34C759',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
  },
  checkoutButtonText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: 'bold',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  errorText: {
    fontSize: 18,
    color: '#FF3B30',
  },
});

// Export the mini-app
export default EcommerceMiniApp;

// Export cleanup function for proper unmounting
export const cleanup = async () => {
  console.log('🧹 Cleaning up E-commerce mini-app');
  // Perform any necessary cleanup
};
