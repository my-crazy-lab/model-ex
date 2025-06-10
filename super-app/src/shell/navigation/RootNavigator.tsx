/**
 * Root Navigator - Main navigation structure for Super App
 */

import React, { useEffect, useState } from 'react';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createDrawerNavigator } from '@react-navigation/drawer';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { useSelector } from 'react-redux';

// Screens
import { HomeScreen } from '../screens/HomeScreen';
import { ProfileScreen } from '../screens/ProfileScreen';
import { SettingsScreen } from '../screens/SettingsScreen';
import { NotificationsScreen } from '../screens/NotificationsScreen';
import { SearchScreen } from '../screens/SearchScreen';

// Auth screens
import { LoginScreen } from '../screens/auth/LoginScreen';
import { RegisterScreen } from '../screens/auth/RegisterScreen';
import { ForgotPasswordScreen } from '../screens/auth/ForgotPasswordScreen';

// Mini-app screens
import { MiniAppScreen } from '../screens/MiniAppScreen';
import { MiniAppStoreScreen } from '../screens/MiniAppStoreScreen';

// Components
import { CustomTabBar } from '../components/CustomTabBar';
import { DrawerContent } from '../components/DrawerContent';
import { LoadingScreen } from '../components/LoadingScreen';

// Services
import { MiniAppManager } from '@shared/services/MiniAppManager';

// Types
import { RootState } from '../store';
import { MiniAppInfo } from '@shared/types/miniApp';

// Navigation types
export type RootStackParamList = {
  Auth: undefined;
  Main: undefined;
  MiniApp: { miniAppId: string; initialRoute?: string };
  MiniAppStore: undefined;
};

export type AuthStackParamList = {
  Login: undefined;
  Register: undefined;
  ForgotPassword: undefined;
};

export type MainTabParamList = {
  Home: undefined;
  Search: undefined;
  MiniApps: undefined;
  Profile: undefined;
  More: undefined;
};

export type DrawerParamList = {
  MainTabs: undefined;
  Settings: undefined;
  Notifications: undefined;
  Help: undefined;
  About: undefined;
};

// Navigators
const RootStack = createNativeStackNavigator<RootStackParamList>();
const AuthStack = createNativeStackNavigator<AuthStackParamList>();
const MainTab = createBottomTabNavigator<MainTabParamList>();
const Drawer = createDrawerNavigator<DrawerParamList>();

/**
 * Auth Navigator - Handles authentication flow
 */
const AuthNavigator: React.FC = () => {
  return (
    <AuthStack.Navigator
      screenOptions={{
        headerShown: false,
        animation: 'slide_from_right',
      }}
    >
      <AuthStack.Screen name="Login" component={LoginScreen} />
      <AuthStack.Screen name="Register" component={RegisterScreen} />
      <AuthStack.Screen name="ForgotPassword" component={ForgotPasswordScreen} />
    </AuthStack.Navigator>
  );
};

/**
 * Main Tab Navigator - Bottom tab navigation
 */
const MainTabNavigator: React.FC = () => {
  const [miniApps, setMiniApps] = useState<MiniAppInfo[]>([]);

  useEffect(() => {
    loadMiniApps();
  }, []);

  const loadMiniApps = async () => {
    try {
      const availableMiniApps = await MiniAppManager.getAvailableMiniApps();
      setMiniApps(availableMiniApps);
    } catch (error) {
      console.error('Failed to load mini-apps:', error);
    }
  };

  return (
    <MainTab.Navigator
      tabBar={(props) => <CustomTabBar {...props} />}
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarIcon: ({ focused, color, size }) => {
          let iconName: string;

          switch (route.name) {
            case 'Home':
              iconName = 'home';
              break;
            case 'Search':
              iconName = 'search';
              break;
            case 'MiniApps':
              iconName = 'apps';
              break;
            case 'Profile':
              iconName = 'person';
              break;
            case 'More':
              iconName = 'more-horiz';
              break;
            default:
              iconName = 'help';
          }

          return <Icon name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: '#007AFF',
        tabBarInactiveTintColor: '#8E8E93',
        tabBarStyle: {
          backgroundColor: '#FFFFFF',
          borderTopWidth: 1,
          borderTopColor: '#E5E5EA',
          paddingBottom: 5,
          paddingTop: 5,
          height: 60,
        },
      })}
    >
      <MainTab.Screen 
        name="Home" 
        component={HomeScreen}
        options={{ title: 'Home' }}
      />
      <MainTab.Screen 
        name="Search" 
        component={SearchScreen}
        options={{ title: 'Search' }}
      />
      <MainTab.Screen 
        name="MiniApps" 
        component={MiniAppStoreScreen}
        options={{ title: 'Apps' }}
      />
      <MainTab.Screen 
        name="Profile" 
        component={ProfileScreen}
        options={{ title: 'Profile' }}
      />
      <MainTab.Screen 
        name="More" 
        component={DrawerNavigator}
        options={{ title: 'More' }}
      />
    </MainTab.Navigator>
  );
};

/**
 * Drawer Navigator - Side menu navigation
 */
const DrawerNavigator: React.FC = () => {
  return (
    <Drawer.Navigator
      drawerContent={(props) => <DrawerContent {...props} />}
      screenOptions={{
        headerShown: true,
        drawerStyle: {
          backgroundColor: '#FFFFFF',
          width: 280,
        },
        drawerActiveTintColor: '#007AFF',
        drawerInactiveTintColor: '#8E8E93',
        drawerLabelStyle: {
          fontSize: 16,
          fontWeight: '500',
        },
      }}
    >
      <Drawer.Screen 
        name="MainTabs" 
        component={MainTabNavigator}
        options={{
          title: 'Super App',
          drawerLabel: 'Home',
          drawerIcon: ({ color, size }) => (
            <Icon name="home" size={size} color={color} />
          ),
        }}
      />
      <Drawer.Screen 
        name="Settings" 
        component={SettingsScreen}
        options={{
          title: 'Settings',
          drawerIcon: ({ color, size }) => (
            <Icon name="settings" size={size} color={color} />
          ),
        }}
      />
      <Drawer.Screen 
        name="Notifications" 
        component={NotificationsScreen}
        options={{
          title: 'Notifications',
          drawerIcon: ({ color, size }) => (
            <Icon name="notifications" size={size} color={color} />
          ),
        }}
      />
    </Drawer.Navigator>
  );
};

/**
 * Main Navigator - Handles authenticated user flow
 */
const MainNavigator: React.FC = () => {
  return <DrawerNavigator />;
};

/**
 * Root Navigator - Top-level navigation
 */
export const RootNavigator: React.FC = () => {
  const { isAuthenticated, isLoading } = useSelector((state: RootState) => state.auth);
  const [isInitializing, setIsInitializing] = useState(true);

  useEffect(() => {
    // Simulate initialization delay
    const timer = setTimeout(() => {
      setIsInitializing(false);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  if (isLoading || isInitializing) {
    return <LoadingScreen message="Loading..." />;
  }

  return (
    <RootStack.Navigator
      screenOptions={{
        headerShown: false,
        animation: 'fade',
      }}
    >
      {isAuthenticated ? (
        <>
          <RootStack.Screen name="Main" component={MainNavigator} />
          <RootStack.Screen 
            name="MiniApp" 
            component={MiniAppScreen}
            options={{
              presentation: 'modal',
              animation: 'slide_from_bottom',
            }}
          />
          <RootStack.Screen 
            name="MiniAppStore" 
            component={MiniAppStoreScreen}
            options={{
              presentation: 'modal',
              animation: 'slide_from_right',
            }}
          />
        </>
      ) : (
        <RootStack.Screen name="Auth" component={AuthNavigator} />
      )}
    </RootStack.Navigator>
  );
};

/**
 * Navigation configuration for deep linking
 */
export const navigationConfig = {
  screens: {
    Auth: {
      screens: {
        Login: 'login',
        Register: 'register',
        ForgotPassword: 'forgot-password',
      },
    },
    Main: {
      screens: {
        MainTabs: {
          screens: {
            Home: 'home',
            Search: 'search',
            MiniApps: 'apps',
            Profile: 'profile',
            More: {
              screens: {
                Settings: 'settings',
                Notifications: 'notifications',
              },
            },
          },
        },
      },
    },
    MiniApp: {
      path: '/mini-app/:miniAppId',
      parse: {
        miniAppId: (miniAppId: string) => miniAppId,
      },
    },
    MiniAppStore: 'app-store',
  },
};

/**
 * Linking configuration
 */
export const linkingConfig = {
  prefixes: ['superapp://', 'https://superapp.com'],
  config: navigationConfig,
};

export default RootNavigator;
