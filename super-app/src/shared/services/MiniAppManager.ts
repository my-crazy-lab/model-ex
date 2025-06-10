/**
 * Mini-App Manager Service
 * 
 * Handles dynamic loading, lifecycle management, and communication
 * between mini-apps and the host application
 */

import { NativeModules, DeviceEventEmitter } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { MiniAppInfo, MiniAppStatus, MiniAppConfig } from '@shared/types/miniApp';
import { AnalyticsService } from './AnalyticsService';
import { SecurityService } from './SecurityService';

/**
 * Mini-App Manager Class
 */
class MiniAppManagerService {
  private loadedMiniApps: Map<string, any> = new Map();
  private miniAppConfigs: Map<string, MiniAppConfig> = new Map();
  private eventListeners: Map<string, Function[]> = new Map();
  private isInitialized = false;

  /**
   * Initialize the Mini-App Manager
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) {
      return;
    }

    console.log('🔧 Initializing Mini-App Manager...');

    try {
      // Load mini-app configurations
      await this.loadMiniAppConfigs();

      // Setup event listeners
      this.setupEventListeners();

      // Preload critical mini-apps
      await this.preloadCriticalMiniApps();

      this.isInitialized = true;
      console.log('✅ Mini-App Manager initialized successfully');

      // Track initialization
      AnalyticsService.trackEvent('miniapp_manager_initialized', {
        loadedMiniApps: this.loadedMiniApps.size,
        availableConfigs: this.miniAppConfigs.size,
      });

    } catch (error) {
      console.error('❌ Failed to initialize Mini-App Manager:', error);
      throw error;
    }
  }

  /**
   * Load mini-app configurations
   */
  private async loadMiniAppConfigs(): Promise<void> {
    try {
      // Load from local storage
      const storedConfigs = await AsyncStorage.getItem('miniapp_configs');
      if (storedConfigs) {
        const configs = JSON.parse(storedConfigs);
        configs.forEach((config: MiniAppConfig) => {
          this.miniAppConfigs.set(config.id, config);
        });
      }

      // Load default configurations
      const defaultConfigs = this.getDefaultMiniAppConfigs();
      defaultConfigs.forEach(config => {
        if (!this.miniAppConfigs.has(config.id)) {
          this.miniAppConfigs.set(config.id, config);
        }
      });

      console.log(`📱 Loaded ${this.miniAppConfigs.size} mini-app configurations`);

    } catch (error) {
      console.error('Failed to load mini-app configurations:', error);
      throw error;
    }
  }

  /**
   * Get default mini-app configurations
   */
  private getDefaultMiniAppConfigs(): MiniAppConfig[] {
    return [
      {
        id: 'ecommerce',
        name: 'E-commerce',
        version: '1.0.0',
        description: 'Shop for products and manage orders',
        icon: 'shopping-cart',
        remoteUrl: 'http://localhost:9001/remoteEntry.js',
        moduleName: 'EcommerceMiniApp',
        exposedModule: './MiniApp',
        permissions: ['camera', 'location', 'storage'],
        status: MiniAppStatus.AVAILABLE,
        category: 'Shopping',
        developer: 'Super App Team',
        rating: 4.5,
        downloadCount: 10000,
        size: '2.5 MB',
        lastUpdated: new Date().toISOString(),
        isPreloadEnabled: true,
        isCritical: false,
      },
      {
        id: 'social',
        name: 'Social Media',
        version: '1.0.0',
        description: 'Connect with friends and share moments',
        icon: 'people',
        remoteUrl: 'http://localhost:9002/remoteEntry.js',
        moduleName: 'SocialMiniApp',
        exposedModule: './MiniApp',
        permissions: ['camera', 'contacts', 'storage'],
        status: MiniAppStatus.AVAILABLE,
        category: 'Social',
        developer: 'Super App Team',
        rating: 4.3,
        downloadCount: 15000,
        size: '3.1 MB',
        lastUpdated: new Date().toISOString(),
        isPreloadEnabled: false,
        isCritical: false,
      },
      {
        id: 'financial',
        name: 'Financial Services',
        version: '1.0.0',
        description: 'Manage your finances and make payments',
        icon: 'account-balance',
        remoteUrl: 'http://localhost:9003/remoteEntry.js',
        moduleName: 'FinancialMiniApp',
        exposedModule: './MiniApp',
        permissions: ['biometric', 'secure-storage'],
        status: MiniAppStatus.AVAILABLE,
        category: 'Finance',
        developer: 'Super App Team',
        rating: 4.7,
        downloadCount: 8000,
        size: '1.8 MB',
        lastUpdated: new Date().toISOString(),
        isPreloadEnabled: true,
        isCritical: true,
      },
      {
        id: 'food-delivery',
        name: 'Food Delivery',
        version: '1.0.0',
        description: 'Order food from your favorite restaurants',
        icon: 'restaurant',
        remoteUrl: 'http://localhost:9004/remoteEntry.js',
        moduleName: 'FoodDeliveryMiniApp',
        exposedModule: './MiniApp',
        permissions: ['location', 'camera'],
        status: MiniAppStatus.AVAILABLE,
        category: 'Food & Drink',
        developer: 'Super App Team',
        rating: 4.4,
        downloadCount: 12000,
        size: '2.8 MB',
        lastUpdated: new Date().toISOString(),
        isPreloadEnabled: false,
        isCritical: false,
      },
    ];
  }

  /**
   * Setup event listeners
   */
  private setupEventListeners(): void {
    // Listen for mini-app events
    DeviceEventEmitter.addListener('miniapp_event', this.handleMiniAppEvent.bind(this));
    DeviceEventEmitter.addListener('miniapp_error', this.handleMiniAppError.bind(this));
  }

  /**
   * Preload critical mini-apps
   */
  private async preloadCriticalMiniApps(): Promise<void> {
    const criticalMiniApps = Array.from(this.miniAppConfigs.values())
      .filter(config => config.isCritical || config.isPreloadEnabled);

    console.log(`🚀 Preloading ${criticalMiniApps.length} critical mini-apps...`);

    const preloadPromises = criticalMiniApps.map(config => 
      this.loadMiniApp(config.id, { preload: true })
    );

    try {
      await Promise.allSettled(preloadPromises);
      console.log('✅ Critical mini-apps preloaded');
    } catch (error) {
      console.warn('⚠️ Some critical mini-apps failed to preload:', error);
    }
  }

  /**
   * Load a mini-app dynamically
   */
  async loadMiniApp(miniAppId: string, options: { preload?: boolean } = {}): Promise<any> {
    try {
      console.log(`📱 Loading mini-app: ${miniAppId}`);

      // Check if already loaded
      if (this.loadedMiniApps.has(miniAppId)) {
        console.log(`✅ Mini-app ${miniAppId} already loaded`);
        return this.loadedMiniApps.get(miniAppId);
      }

      // Get configuration
      const config = this.miniAppConfigs.get(miniAppId);
      if (!config) {
        throw new Error(`Mini-app configuration not found: ${miniAppId}`);
      }

      // Check permissions
      await this.checkPermissions(config.permissions);

      // Security validation
      await SecurityService.validateMiniApp(config);

      // Track loading start
      const loadStartTime = Date.now();
      AnalyticsService.trackEvent('miniapp_load_start', {
        miniAppId,
        isPreload: options.preload || false,
      });

      // Dynamic import using module federation
      const miniApp = await this.dynamicImport(config);

      // Cache the loaded mini-app
      this.loadedMiniApps.set(miniAppId, miniApp);

      // Track loading success
      const loadTime = Date.now() - loadStartTime;
      AnalyticsService.trackEvent('miniapp_load_success', {
        miniAppId,
        loadTime,
        isPreload: options.preload || false,
      });

      console.log(`✅ Mini-app ${miniAppId} loaded successfully in ${loadTime}ms`);

      // Emit load event
      this.emitEvent('miniapp_loaded', { miniAppId, loadTime });

      return miniApp;

    } catch (error) {
      console.error(`❌ Failed to load mini-app ${miniAppId}:`, error);

      // Track loading error
      AnalyticsService.trackError('miniapp_load_failed', error, {
        miniAppId,
        isPreload: options.preload || false,
      });

      throw error;
    }
  }

  /**
   * Dynamic import using module federation
   */
  private async dynamicImport(config: MiniAppConfig): Promise<any> {
    try {
      // In a real implementation, this would use Webpack Module Federation
      // For now, we'll simulate the dynamic import
      
      if (__DEV__) {
        // Development mode - load from local modules
        switch (config.id) {
          case 'ecommerce':
            return await import('../../mini-apps/ecommerce');
          case 'social':
            return await import('../../mini-apps/social');
          case 'financial':
            return await import('../../mini-apps/financial');
          case 'food-delivery':
            return await import('../../mini-apps/food-delivery');
          default:
            throw new Error(`Unknown mini-app: ${config.id}`);
        }
      } else {
        // Production mode - load from remote URLs
        const remoteModule = await this.loadRemoteModule(config.remoteUrl, config.moduleName);
        return remoteModule[config.exposedModule];
      }

    } catch (error) {
      console.error('Dynamic import failed:', error);
      throw error;
    }
  }

  /**
   * Load remote module (production)
   */
  private async loadRemoteModule(remoteUrl: string, moduleName: string): Promise<any> {
    // This would be implemented using Webpack Module Federation
    // For now, we'll throw an error to indicate it's not implemented
    throw new Error('Remote module loading not implemented in this example');
  }

  /**
   * Check required permissions
   */
  private async checkPermissions(permissions: string[]): Promise<void> {
    // Implementation would check and request permissions
    console.log('Checking permissions:', permissions);
    // For now, we'll assume all permissions are granted
  }

  /**
   * Unload a mini-app
   */
  async unloadMiniApp(miniAppId: string): Promise<void> {
    try {
      console.log(`🗑️ Unloading mini-app: ${miniAppId}`);

      if (!this.loadedMiniApps.has(miniAppId)) {
        console.log(`Mini-app ${miniAppId} is not loaded`);
        return;
      }

      // Get the mini-app instance
      const miniApp = this.loadedMiniApps.get(miniAppId);

      // Call cleanup if available
      if (miniApp && typeof miniApp.cleanup === 'function') {
        await miniApp.cleanup();
      }

      // Remove from cache
      this.loadedMiniApps.delete(miniAppId);

      // Track unloading
      AnalyticsService.trackEvent('miniapp_unloaded', { miniAppId });

      // Emit unload event
      this.emitEvent('miniapp_unloaded', { miniAppId });

      console.log(`✅ Mini-app ${miniAppId} unloaded successfully`);

    } catch (error) {
      console.error(`❌ Failed to unload mini-app ${miniAppId}:`, error);
      throw error;
    }
  }

  /**
   * Get available mini-apps
   */
  async getAvailableMiniApps(): Promise<MiniAppInfo[]> {
    return Array.from(this.miniAppConfigs.values()).map(config => ({
      id: config.id,
      name: config.name,
      version: config.version,
      description: config.description,
      icon: config.icon,
      status: config.status,
      category: config.category,
      developer: config.developer,
      rating: config.rating,
      downloadCount: config.downloadCount,
      size: config.size,
      lastUpdated: config.lastUpdated,
      isLoaded: this.loadedMiniApps.has(config.id),
    }));
  }

  /**
   * Get loaded mini-apps
   */
  getLoadedMiniApps(): string[] {
    return Array.from(this.loadedMiniApps.keys());
  }

  /**
   * Check if mini-app is loaded
   */
  isMiniAppLoaded(miniAppId: string): boolean {
    return this.loadedMiniApps.has(miniAppId);
  }

  /**
   * Get mini-app configuration
   */
  getMiniAppConfig(miniAppId: string): MiniAppConfig | undefined {
    return this.miniAppConfigs.get(miniAppId);
  }

  /**
   * Update mini-app configuration
   */
  async updateMiniAppConfig(miniAppId: string, updates: Partial<MiniAppConfig>): Promise<void> {
    const config = this.miniAppConfigs.get(miniAppId);
    if (!config) {
      throw new Error(`Mini-app configuration not found: ${miniAppId}`);
    }

    const updatedConfig = { ...config, ...updates };
    this.miniAppConfigs.set(miniAppId, updatedConfig);

    // Save to storage
    await this.saveMiniAppConfigs();

    // Track update
    AnalyticsService.trackEvent('miniapp_config_updated', { miniAppId, updates });
  }

  /**
   * Save mini-app configurations to storage
   */
  private async saveMiniAppConfigs(): Promise<void> {
    try {
      const configs = Array.from(this.miniAppConfigs.values());
      await AsyncStorage.setItem('miniapp_configs', JSON.stringify(configs));
    } catch (error) {
      console.error('Failed to save mini-app configurations:', error);
    }
  }

  /**
   * Handle mini-app events
   */
  private handleMiniAppEvent(event: any): void {
    console.log('Mini-app event received:', event);
    
    // Track event
    AnalyticsService.trackEvent('miniapp_event_received', event);

    // Emit to listeners
    this.emitEvent(event.type, event.data);
  }

  /**
   * Handle mini-app errors
   */
  private handleMiniAppError(error: any): void {
    console.error('Mini-app error:', error);
    
    // Track error
    AnalyticsService.trackError('miniapp_error', error);

    // Emit error event
    this.emitEvent('miniapp_error', error);
  }

  /**
   * Add event listener
   */
  addEventListener(eventType: string, listener: Function): void {
    if (!this.eventListeners.has(eventType)) {
      this.eventListeners.set(eventType, []);
    }
    this.eventListeners.get(eventType)!.push(listener);
  }

  /**
   * Remove event listener
   */
  removeEventListener(eventType: string, listener: Function): void {
    const listeners = this.eventListeners.get(eventType);
    if (listeners) {
      const index = listeners.indexOf(listener);
      if (index > -1) {
        listeners.splice(index, 1);
      }
    }
  }

  /**
   * Emit event to listeners
   */
  private emitEvent(eventType: string, data: any): void {
    const listeners = this.eventListeners.get(eventType);
    if (listeners) {
      listeners.forEach(listener => {
        try {
          listener(data);
        } catch (error) {
          console.error('Error in event listener:', error);
        }
      });
    }
  }

  /**
   * Clear all loaded mini-apps
   */
  async clearAll(): Promise<void> {
    const loadedMiniAppIds = Array.from(this.loadedMiniApps.keys());
    
    for (const miniAppId of loadedMiniAppIds) {
      try {
        await this.unloadMiniApp(miniAppId);
      } catch (error) {
        console.error(`Failed to unload mini-app ${miniAppId}:`, error);
      }
    }

    console.log('✅ All mini-apps cleared');
  }
}

// Export singleton instance
export const MiniAppManager = new MiniAppManagerService();
