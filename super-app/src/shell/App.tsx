/**
 * Super App - Main Application Shell
 * 
 * This is the host application that manages mini-apps through module federation
 */

import React, { useEffect, useState } from 'react';
import {
  StatusBar,
  StyleSheet,
  View,
  Alert,
  AppState,
  AppStateStatus,
} from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { Provider } from 'react-redux';
import { PersistGate } from 'redux-persist/integration/react';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { enableScreens } from 'react-native-screens';
import SplashScreen from 'react-native-bootsplash';

// Shell components
import { RootNavigator } from './navigation/RootNavigator';
import { LoadingScreen } from './components/LoadingScreen';
import { ErrorBoundary } from './components/ErrorBoundary';
import { NetworkStatusProvider } from './providers/NetworkStatusProvider';
import { AuthProvider } from './providers/AuthProvider';
import { MiniAppProvider } from './providers/MiniAppProvider';
import { ThemeProvider } from './providers/ThemeProvider';

// Store
import { store, persistor } from './store';

// Services
import { AuthService } from '@shared/services/AuthService';
import { MiniAppManager } from '@shared/services/MiniAppManager';
import { AnalyticsService } from '@shared/services/AnalyticsService';
import { UpdateService } from '@shared/services/UpdateService';
import { SecurityService } from '@shared/services/SecurityService';

// Utils
import { navigationRef } from './navigation/NavigationService';
import { setupInterceptors } from '@shared/utils/apiInterceptors';
import { initializeCrashReporting } from '@shared/utils/crashReporting';

// Types
import { AppConfig } from '@shared/types/app';

// Enable screens for better performance
enableScreens();

/**
 * Main App Component
 */
const App: React.FC = () => {
  const [isInitialized, setIsInitialized] = useState(false);
  const [initializationError, setInitializationError] = useState<string | null>(null);

  useEffect(() => {
    initializeApp();
  }, []);

  useEffect(() => {
    const handleAppStateChange = (nextAppState: AppStateStatus) => {
      if (nextAppState === 'active') {
        // App came to foreground
        AnalyticsService.trackEvent('app_foreground');
        SecurityService.checkAppIntegrity();
      } else if (nextAppState === 'background') {
        // App went to background
        AnalyticsService.trackEvent('app_background');
        SecurityService.lockSensitiveData();
      }
    };

    const subscription = AppState.addEventListener('change', handleAppStateChange);
    return () => subscription?.remove();
  }, []);

  /**
   * Initialize the application
   */
  const initializeApp = async () => {
    try {
      console.log('🚀 Initializing Super App...');

      // Initialize crash reporting first
      await initializeCrashReporting();

      // Initialize security services
      await SecurityService.initialize();

      // Setup API interceptors
      setupInterceptors();

      // Initialize analytics
      await AnalyticsService.initialize();

      // Initialize authentication service
      await AuthService.initialize();

      // Initialize mini-app manager
      await MiniAppManager.initialize();

      // Check for app updates
      await UpdateService.checkForUpdates();

      // Track app launch
      AnalyticsService.trackEvent('app_launch', {
        timestamp: new Date().toISOString(),
        version: AppConfig.version,
      });

      console.log('✅ Super App initialized successfully');
      setIsInitialized(true);

      // Hide splash screen
      await SplashScreen.hide({ fade: true });

    } catch (error) {
      console.error('❌ Failed to initialize Super App:', error);
      setInitializationError(error instanceof Error ? error.message : 'Unknown error');
      
      // Track initialization error
      AnalyticsService.trackError('app_initialization_failed', error);

      // Show error alert
      Alert.alert(
        'Initialization Error',
        'Failed to initialize the app. Please restart the application.',
        [
          {
            text: 'Retry',
            onPress: () => {
              setInitializationError(null);
              initializeApp();
            },
          },
          {
            text: 'Exit',
            onPress: () => {
              // In a real app, you might want to close the app
              console.log('User chose to exit');
            },
          },
        ]
      );
    }
  };

  /**
   * Handle navigation ready
   */
  const onNavigationReady = () => {
    console.log('📱 Navigation ready');
    AnalyticsService.trackEvent('navigation_ready');
  };

  /**
   * Handle navigation state change
   */
  const onNavigationStateChange = (state: any) => {
    // Track navigation events
    const currentRoute = state?.routes?.[state.index];
    if (currentRoute) {
      AnalyticsService.trackScreenView(currentRoute.name, currentRoute.params);
    }
  };

  // Show loading screen during initialization
  if (!isInitialized && !initializationError) {
    return <LoadingScreen message="Initializing Super App..." />;
  }

  // Show error screen if initialization failed
  if (initializationError) {
    return (
      <ErrorBoundary
        error={new Error(initializationError)}
        onRetry={() => {
          setInitializationError(null);
          initializeApp();
        }}
      />
    );
  }

  return (
    <ErrorBoundary>
      <GestureHandlerRootView style={styles.container}>
        <SafeAreaProvider>
          <Provider store={store}>
            <PersistGate loading={<LoadingScreen />} persistor={persistor}>
              <ThemeProvider>
                <NetworkStatusProvider>
                  <AuthProvider>
                    <MiniAppProvider>
                      <StatusBar
                        barStyle="dark-content"
                        backgroundColor="transparent"
                        translucent
                      />
                      <NavigationContainer
                        ref={navigationRef}
                        onReady={onNavigationReady}
                        onStateChange={onNavigationStateChange}
                      >
                        <RootNavigator />
                      </NavigationContainer>
                    </MiniAppProvider>
                  </AuthProvider>
                </NetworkStatusProvider>
              </ThemeProvider>
            </PersistGate>
          </Provider>
        </SafeAreaProvider>
      </GestureHandlerRootView>
    </ErrorBoundary>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});

export default App;
