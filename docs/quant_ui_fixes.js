/**
 * Quant Dashboard - UI Component Fixes
 * =====================================
 * Các components đã được fix theo QA report.
 */

// =============================================================================
// 1. NUMBER FORMATTING UTILITIES (Fix H3)
// =============================================================================

export const formatters = {
  /**
   * Format giá tiền: 1,234.56
   */
  price: (value, decimals = 2) => {
    if (value == null || isNaN(value)) return '-';
    return value.toLocaleString('en-US', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  },

  /**
   * Format phần trăm: 12.34%
   * @param {number} value - Giá trị (0.1234 sẽ thành 12.34%)
   * @param {boolean} isAlreadyPercent - True nếu value đã là %, ví dụ 12.34
   */
  percent: (value, decimals = 2, isAlreadyPercent = false) => {
    if (value == null || isNaN(value)) return '-';
    const displayValue = isAlreadyPercent ? value : value * 100;
    return `${displayValue.toFixed(decimals)}%`;
  },

  /**
   * Format ratio: 0.1234
   */
  ratio: (value, decimals = 4) => {
    if (value == null || isNaN(value)) return '-';
    return value.toFixed(decimals);
  },

  /**
   * Format volume với suffix K/M/B
   */
  volume: (value) => {
    if (value == null || isNaN(value)) return '-';
    if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
    if (value >= 1e3) return `${(value / 1e3).toFixed(2)}K`;
    return value.toLocaleString('en-US', { maximumFractionDigits: 0 });
  },

  /**
   * Format Sharpe ratio: 1.23
   */
  sharpe: (value) => {
    if (value == null || isNaN(value)) return '-';
    return value.toFixed(2);
  },

  /**
   * Format ngày theo locale Việt Nam
   */
  date: (value) => {
    if (!value) return '-';
    return new Date(value).toLocaleDateString('vi-VN');
  },

  /**
   * Format datetime đầy đủ
   */
  datetime: (value) => {
    if (!value) return '-';
    return new Date(value).toLocaleString('vi-VN');
  },
};


// =============================================================================
// 2. DATA VALIDATION (Fix C1, C3) - Frontend validation layer
// =============================================================================

export const validators = {
  /**
   * Validate price hợp lệ
   */
  isValidPrice: (price) => {
    return price != null && price >= 0.01 && price <= 1000000;
  },

  /**
   * Validate daily return hợp lệ
   */
  isValidDailyReturn: (ret) => {
    return ret != null && Math.abs(ret) <= 1;
  },

  /**
   * Validate Sharpe ratio hợp lệ
   */
  isValidSharpe: (sharpe) => {
    return sharpe != null && sharpe >= -5 && sharpe <= 10;
  },

  /**
   * Validate max drawdown hợp lệ
   */
  isValidDrawdown: (dd) => {
    return dd != null && dd <= 0 && dd >= -100;
  },

  /**
   * Highlight cell nếu giá trị không hợp lệ
   */
  getCellStyle: (value, validator) => {
    if (!validator(value)) {
      return {
        backgroundColor: '#ffcccc',
        color: '#c00000',
        fontWeight: 'bold',
      };
    }
    return {};
  },
};


// =============================================================================
// 3. i18n TRANSLATIONS (Fix C2)
// =============================================================================

export const translations = {
  vi: {
    // Navigation
    nav: {
      overview: 'Tổng quan',
      risk_metrics: 'Chỉ số Rủi ro',
      sector_analysis: 'Phân tích Ngành',
      investment_strategy: 'Chiến lược Đầu tư',
      ml_models: 'Mô hình ML',
      settings: 'Cài đặt',
    },

    // Headers
    headers: {
      portfolio_overview: 'Tổng quan Danh mục',
      sharpe_distribution: 'Phân phối Tỷ lệ Sharpe',
      sector_performance: 'Hiệu suất theo Ngành',
      risk_return_profile: 'Hồ sơ Rủi ro - Lợi nhuận',
      top_by_sharpe: 'Top 10 theo Tỷ lệ Sharpe',
      strategy_results: 'Kết quả Chiến lược',
      cumulative_return: 'Lợi nhuận Tích lũy',
      strategy_details: 'Chi tiết Chiến lược',
      ml_models: 'Mô hình Học máy',
      causal_analysis: 'Phân tích Nhân quả',
      feature_importance: 'Độ quan trọng Đặc trưng',
    },

    // Metrics
    metrics: {
      total_tickers: 'Tổng số Mã',
      sharpe_ratio: 'Tỷ lệ Sharpe',
      volatility: 'Độ biến động',
      max_drawdown: 'Sụt giảm Tối đa',
      portfolio: 'Danh mục',
      holdings: 'Số Mã Nắm giữ',
      beta: 'Hệ số Beta',
      return_stability: 'Độ ổn định Lợi nhuận',
      avg_momentum: 'Đà tăng Trung bình',
    },

    // Strategies
    strategies: {
      low_beta_quality: 'Beta thấp & Chất lượng',
      sector_rotation: 'Luân chuyển Ngành',
      momentum: 'Đà tăng giá',
      sentiment: 'Tâm lý Thị trường',
    },

    // Table headers
    table: {
      ticker: 'Mã',
      sector: 'Ngành',
      num_records: 'Số bản ghi',
      first_date: 'Ngày đầu',
      last_date: 'Ngày cuối',
      first_price: 'Giá đầu',
      last_price: 'Giá cuối',
      price_change: 'Biến động giá',
      daily_return: 'Lợi nhuận TB ngày',
      volatility: 'Độ biến động',
      sharpe_ratio: 'Tỷ lệ Sharpe',
      max_drawdown: 'Sụt giảm Tối đa',
      avg_volume: 'Khối lượng TB',
    },

    // Actions
    actions: {
      run_backtest: 'Chạy Kiểm định',
      clear_cache: 'Xóa Cache',
      export_data: 'Xuất Dữ liệu',
      view_details: 'Xem Chi tiết',
    },

    // Status
    status: {
      online: 'Trực tuyến',
      connected: 'Đã kết nối',
      loading: 'Đang tải...',
      updated: 'Vừa cập nhật',
    },

    // Time periods
    time: {
      all_time: 'Toàn bộ thời gian',
      last_30_days: '30 ngày gần đây',
      start_date: 'Ngày bắt đầu',
      end_date: 'Ngày kết thúc',
    },
  },
};

/**
 * Hook để sử dụng translations
 * @example
 * const t = useTranslation('vi');
 * console.log(t.nav.overview); // "Tổng quan"
 */
export const useTranslation = (locale = 'vi') => {
  return translations[locale] || translations.vi;
};


// =============================================================================
// 4. IMPROVED CHART CONFIGS (Fix H1, H2, C4)
// =============================================================================

/**
 * Config cho Scatter plot với nhiều data points
 * Fix H1: Scatter plot 100 ticker quá đông
 */
export const scatterPlotConfig = {
  // Giảm opacity để dễ nhìn khi có nhiều points
  pointOpacity: 0.6,
  pointSize: 8,
  
  // Enable zoom và pan
  zoom: {
    enabled: true,
    mode: 'xy',
    wheel: { enabled: true },
    pinch: { enabled: true },
  },
  
  pan: {
    enabled: true,
    mode: 'xy',
  },
  
  // Tooltip với đầy đủ thông tin
  tooltip: {
    enabled: true,
    callbacks: {
      label: (context) => {
        const point = context.raw;
        return [
          `Ticker: ${point.ticker}`,
          `Sharpe: ${formatters.sharpe(point.sharpe)}`,
          `Volatility: ${formatters.percent(point.volatility, 2, true)}`,
          `Sector: ${point.sector}`,
        ];
      },
    },
  },
};

/**
 * Config cho Bar chart với labels dài
 * Fix H2: Bar chart labels bị nghiêng 45° khó đọc
 */
export const horizontalBarConfig = {
  indexAxis: 'y', // Horizontal bars
  responsive: true,
  maintainAspectRatio: false,
  
  plugins: {
    legend: { display: false },
  },
  
  scales: {
    x: {
      beginAtZero: true,
      title: {
        display: true,
        text: 'Sharpe Ratio',
      },
    },
    y: {
      ticks: {
        // Không cần rotate khi dùng horizontal bar
        maxRotation: 0,
        autoSkip: false,
      },
    },
  },
};

/**
 * Config cho Pie chart với nhiều segments
 * Fix C4: Pie chart không readable với 30 holdings
 * Solution: Group nhỏ thành "Others" và dùng bar chart
 */
export const groupSmallSegments = (data, threshold = 0.02) => {
  // Sort by value descending
  const sorted = [...data].sort((a, b) => b.value - a.value);
  const total = sorted.reduce((sum, item) => sum + item.value, 0);
  
  const significant = [];
  let othersValue = 0;
  
  for (const item of sorted) {
    if (item.value / total >= threshold) {
      significant.push(item);
    } else {
      othersValue += item.value;
    }
  }
  
  if (othersValue > 0) {
    significant.push({
      name: 'Khác',
      value: othersValue,
      isOthers: true,
    });
  }
  
  return significant;
};


// =============================================================================
// 5. LOADING STATE COMPONENT (Fix H4)
// =============================================================================

/**
 * Skeleton loader cho tables và charts
 */
export const SkeletonLoader = ({ type = 'table', rows = 5 }) => {
  if (type === 'table') {
    return (
      <div className="animate-pulse">
        {/* Header */}
        <div className="h-10 bg-gray-700 rounded mb-2" />
        {/* Rows */}
        {Array.from({ length: rows }).map((_, i) => (
          <div key={i} className="h-8 bg-gray-800 rounded mb-1" />
        ))}
      </div>
    );
  }
  
  if (type === 'chart') {
    return (
      <div className="animate-pulse">
        <div className="h-64 bg-gray-800 rounded" />
      </div>
    );
  }
  
  return null;
};


// =============================================================================
// 6. TABLE COMPONENT WITH SORTING & PAGINATION (Fix M1, M2, M3)
// =============================================================================

/**
 * Enhanced table component
 */
export const DataTable = ({
  data,
  columns,
  pageSize = 20,
  stickyHeader = true,
  sortable = true,
}) => {
  const [currentPage, setCurrentPage] = React.useState(1);
  const [sortConfig, setSortConfig] = React.useState({ key: null, direction: 'asc' });
  
  // Sorting logic
  const sortedData = React.useMemo(() => {
    if (!sortConfig.key) return data;
    
    return [...data].sort((a, b) => {
      if (a[sortConfig.key] < b[sortConfig.key]) {
        return sortConfig.direction === 'asc' ? -1 : 1;
      }
      if (a[sortConfig.key] > b[sortConfig.key]) {
        return sortConfig.direction === 'asc' ? 1 : -1;
      }
      return 0;
    });
  }, [data, sortConfig]);
  
  // Pagination logic
  const totalPages = Math.ceil(sortedData.length / pageSize);
  const paginatedData = sortedData.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize
  );
  
  const handleSort = (key) => {
    if (!sortable) return;
    
    setSortConfig((prev) => ({
      key,
      direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc',
    }));
  };
  
  const getSortIndicator = (key) => {
    if (sortConfig.key !== key) return '⇅';
    return sortConfig.direction === 'asc' ? '▲' : '▼';
  };
  
  return (
    <div className="overflow-auto">
      <table className="w-full border-collapse">
        <thead className={stickyHeader ? 'sticky top-0 z-10' : ''}>
          <tr className="bg-gray-800">
            {columns.map((col) => (
              <th
                key={col.key}
                onClick={() => handleSort(col.key)}
                className={`
                  p-3 text-left text-sm font-semibold text-gray-200
                  ${sortable ? 'cursor-pointer hover:bg-gray-700' : ''}
                `}
              >
                {col.label}
                {sortable && (
                  <span className="ml-2 text-gray-500">
                    {getSortIndicator(col.key)}
                  </span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {paginatedData.map((row, i) => (
            <tr
              key={i}
              className="border-b border-gray-700 hover:bg-gray-800"
            >
              {columns.map((col) => (
                <td key={col.key} className="p-3 text-sm">
                  {col.render ? col.render(row[col.key], row) : row[col.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      
      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between p-4 border-t border-gray-700">
          <span className="text-sm text-gray-400">
            Hiển thị {(currentPage - 1) * pageSize + 1} - {Math.min(currentPage * pageSize, sortedData.length)} / {sortedData.length}
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="px-3 py-1 bg-gray-700 rounded disabled:opacity-50"
            >
              ←
            </button>
            <span className="px-3 py-1">
              {currentPage} / {totalPages}
            </span>
            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              className="px-3 py-1 bg-gray-700 rounded disabled:opacity-50"
            >
              →
            </button>
          </div>
        </div>
      )}
    </div>
  );
};


// =============================================================================
// 7. COLOR SCHEME FIXES (Fix H6)
// =============================================================================

/**
 * Màu sắc theo convention đúng:
 * - Xanh lá = Tích cực, Có ý nghĩa, Thành công
 * - Đỏ = Tiêu cực, Không có ý nghĩa, Lỗi
 */
export const colors = {
  // Semantic colors
  positive: '#00B050',      // Xanh lá - tích cực
  negative: '#C00000',      // Đỏ - tiêu cực
  neutral: '#808080',       // Xám - trung tính
  
  // Significance colors (cho p-values)
  significant: '#00B050',   // Xanh lá - có ý nghĩa thống kê (p < 0.05)
  notSignificant: '#FFA500', // Cam - không có ý nghĩa thống kê
  
  // Performance colors
  profit: '#00B050',        // Lãi
  loss: '#C00000',          // Lỗ
  
  // Status colors
  online: '#00B050',
  offline: '#C00000',
  warning: '#FFA500',
  
  // Chart colors (mảng cho multiple series)
  chartPalette: [
    '#2E86AB', // Blue
    '#A23B72', // Purple
    '#F18F01', // Orange
    '#C73E1D', // Red
    '#3B1F2B', // Dark
    '#95C623', // Green
  ],
};

/**
 * Helper để lấy màu dựa trên giá trị
 */
export const getColorByValue = (value, threshold = 0) => {
  if (value > threshold) return colors.positive;
  if (value < threshold) return colors.negative;
  return colors.neutral;
};

/**
 * Helper để lấy màu cho p-value
 */
export const getSignificanceColor = (pValue, alpha = 0.05) => {
  return pValue < alpha ? colors.significant : colors.notSignificant;
};
