const path = require('path');
const { RepackPlugin } = require('@callstack/repack');

/**
 * Repack configuration for Super App
 * Enables module federation and micro-frontend architecture
 */
module.exports = {
  webpack: (env) => {
    const {
      mode = 'development',
      context = __dirname,
      entry = './index.js',
      platform,
      minimize = mode === 'production',
      devServer = undefined,
      bundleFilename = undefined,
      sourceMapFilename = undefined,
      assetsPath = undefined,
      reactNativePath = require.resolve('react-native'),
    } = env;

    return {
      mode,
      devtool: mode === 'development' ? 'eval-source-map' : 'source-map',
      context,
      entry: [
        ...RepackPlugin.getInitializationEntries(reactNativePath),
        entry,
      ],
      resolve: {
        alias: {
          'react-native': reactNativePath,
          '@': path.resolve(__dirname, 'src'),
          '@shell': path.resolve(__dirname, 'src/shell'),
          '@mini-apps': path.resolve(__dirname, 'src/mini-apps'),
          '@shared': path.resolve(__dirname, 'src/shared'),
          '@assets': path.resolve(__dirname, 'src/assets'),
        },
        extensions: ['.js', '.jsx', '.ts', '.tsx', '.json'],
      },
      module: {
        rules: [
          {
            test: /\.[jt]sx?$/,
            include: [
              /node_modules(.*[/\\])+react/,
              /node_modules(.*[/\\])+@react-native/,
              /node_modules(.*[/\\])+@react-navigation/,
              /node_modules(.*[/\\])+@reduxjs/,
              context,
            ],
            use: 'babel-loader',
          },
          {
            test: /\.(png|jpe?g|gif|svg)$/,
            use: {
              loader: '@callstack/repack/assets-loader',
              options: {
                platform,
                devServerEnabled: Boolean(devServer),
                scalableAssetExtensions: RepackPlugin.SCALABLE_ASSETS,
              },
            },
          },
        ],
      },
      plugins: [
        new RepackPlugin({
          context,
          mode,
          platform,
          devServer,
          output: {
            bundleFilename,
            sourceMapFilename,
            assetsPath,
          },
          extraChunks: [
            {
              include: /node_modules/,
              type: 'remote',
              outputPath: path.resolve(__dirname, 'build/output'),
            },
          ],
        }),
        // Module Federation Plugin for micro-frontends
        new RepackPlugin.ModuleFederationPlugin({
          name: 'SuperAppHost',
          remotes: {
            EcommerceMiniApp: 'EcommerceMiniApp@http://localhost:9001/remoteEntry.js',
            SocialMiniApp: 'SocialMiniApp@http://localhost:9002/remoteEntry.js',
            FinancialMiniApp: 'FinancialMiniApp@http://localhost:9003/remoteEntry.js',
            FoodDeliveryMiniApp: 'FoodDeliveryMiniApp@http://localhost:9004/remoteEntry.js',
          },
          shared: {
            react: {
              singleton: true,
              eager: true,
              requiredVersion: '^18.2.0',
            },
            'react-native': {
              singleton: true,
              eager: true,
              requiredVersion: '^0.72.0',
            },
            '@react-navigation/native': {
              singleton: true,
              eager: true,
            },
            '@reduxjs/toolkit': {
              singleton: true,
              eager: true,
            },
            'react-redux': {
              singleton: true,
              eager: true,
            },
          },
        }),
      ],
      optimization: {
        minimize,
        chunkIds: 'named',
        splitChunks: {
          chunks: 'all',
          cacheGroups: {
            vendor: {
              test: /[\\/]node_modules[\\/]/,
              name: 'vendors',
              chunks: 'all',
              priority: 10,
            },
            common: {
              name: 'common',
              minChunks: 2,
              chunks: 'all',
              priority: 5,
              reuseExistingChunk: true,
            },
            miniApps: {
              test: /[\\/]src[\\/]mini-apps[\\/]/,
              name: 'mini-apps',
              chunks: 'all',
              priority: 8,
            },
          },
        },
      },
      performance: {
        hints: mode === 'production' ? 'warning' : false,
        maxAssetSize: 300000,
        maxEntrypointSize: 300000,
      },
    };
  },
};

/**
 * Mini-app specific configurations
 */
const createMiniAppConfig = (miniAppName, port) => ({
  webpack: (env) => {
    const baseConfig = module.exports.webpack(env);
    
    return {
      ...baseConfig,
      plugins: [
        ...baseConfig.plugins.filter(plugin => 
          plugin.constructor.name !== 'ModuleFederationPlugin'
        ),
        new RepackPlugin.ModuleFederationPlugin({
          name: miniAppName,
          filename: 'remoteEntry.js',
          exposes: {
            './MiniApp': `./src/mini-apps/${miniAppName.toLowerCase()}/index.tsx`,
          },
          shared: baseConfig.plugins
            .find(plugin => plugin.constructor.name === 'ModuleFederationPlugin')
            ?.options?.shared || {},
        }),
      ],
      devServer: {
        port,
        hot: true,
        liveReload: true,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, PATCH, OPTIONS',
          'Access-Control-Allow-Headers': 'X-Requested-With, content-type, Authorization',
        },
      },
    };
  },
});

// Export mini-app configurations
module.exports.ecommerce = createMiniAppConfig('EcommerceMiniApp', 9001);
module.exports.social = createMiniAppConfig('SocialMiniApp', 9002);
module.exports.financial = createMiniAppConfig('FinancialMiniApp', 9003);
module.exports.foodDelivery = createMiniAppConfig('FoodDeliveryMiniApp', 9004);

/**
 * Development server configuration
 */
module.exports.devServer = {
  port: 8081,
  host: 'localhost',
  hot: true,
  liveReload: true,
  compress: true,
  allowedHosts: 'all',
  headers: {
    'Access-Control-Allow-Origin': '*',
  },
  static: {
    directory: path.join(__dirname, 'build'),
    publicPath: '/build/',
  },
  client: {
    overlay: {
      errors: true,
      warnings: false,
    },
  },
};

/**
 * Production optimization
 */
module.exports.optimization = {
  production: {
    minimize: true,
    sideEffects: false,
    usedExports: true,
    concatenateModules: true,
    splitChunks: {
      chunks: 'all',
      minSize: 20000,
      maxSize: 244000,
      cacheGroups: {
        vendor: {
          test: /[\\/]node_modules[\\/]/,
          name: 'vendors',
          chunks: 'all',
          priority: 10,
        },
        common: {
          name: 'common',
          minChunks: 2,
          chunks: 'all',
          priority: 5,
          reuseExistingChunk: true,
        },
      },
    },
  },
};

/**
 * Environment-specific configurations
 */
module.exports.environments = {
  development: {
    devtool: 'eval-source-map',
    optimization: {
      minimize: false,
    },
  },
  staging: {
    devtool: 'source-map',
    optimization: {
      minimize: true,
    },
  },
  production: {
    devtool: 'hidden-source-map',
    optimization: module.exports.optimization.production,
  },
};
