import React, { useState, useEffect, useCallback } from 'react';
import { 
  BarChart3, 
  TrendingUp, 
  PieChart, 
  Activity, 
  FileText, 
  ArrowLeft, 
  Loader2, 
  AlertCircle,
  Download,
  Database,
  Grid,
  Info
} from 'lucide-react';

const API_BASE_URL = 'http://192.168.190.28:9096';

// Fetch Excel analytics
const fetchExcelAnalytics = async (threadId, documentId) => {
  try {
    const formData = new FormData();
    formData.append('thread_id', threadId);
    formData.append('document_id', documentId);

    const response = await fetch(`${API_BASE_URL}/chat/excel_analytics`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Analytics fetch failed: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error fetching analytics:', error);
    throw error;
  }
};

// Fetch visualization suggestions
const fetchVisualizationSuggestions = async (threadId, documentId) => {
  try {
    const formData = new FormData();
    formData.append('thread_id', threadId);
    formData.append('document_id', documentId);

    const response = await fetch(`${API_BASE_URL}/chat/excel_visualizations`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Visualization suggestions failed: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error fetching visualizations:', error);
    throw error;
  }
};

const ExcelAnalytics = ({ threadId, documentId, documentName, onBack }) => {
  const [analytics, setAnalytics] = useState(null);
  const [visualizations, setVisualizations] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    const loadAnalytics = async () => {
      if (!threadId || !documentId) {
        setError('Missing thread or document ID');
        setIsLoading(false);
        return;
      }

      setIsLoading(true);
      setError(null);

      try {
        const [analyticsData, visualizationData] = await Promise.all([
          fetchExcelAnalytics(threadId, documentId),
          fetchVisualizationSuggestions(threadId, documentId)
        ]);

        setAnalytics(analyticsData);
        setVisualizations(visualizationData);
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };

    loadAnalytics();
  }, [threadId, documentId]);

  const renderOverview = () => {
    if (!analytics) return null;

    return (
      <div className="space-y-6">
        {/* File Info Card */}
        <div className="bg-white dark:bg-zinc-800 rounded-xl shadow-lg p-6 border border-gray-200 dark:border-zinc-700">
          <div className="flex items-center mb-4">
            <FileText className="w-6 h-6 text-blue-500 mr-3" />
            <h3 className="text-xl font-bold text-gray-800 dark:text-gray-100">File Information</h3>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Filename</p>
              <p className="text-base font-semibold text-gray-800 dark:text-gray-100">{analytics.filename}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Total Sheets</p>
              <p className="text-base font-semibold text-gray-800 dark:text-gray-100">{analytics.sheets?.length || 0}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Total Chunks</p>
              <p className="text-base font-semibold text-gray-800 dark:text-gray-100">{analytics.total_chunks}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Analytics Chunks</p>
              <p className="text-base font-semibold text-gray-800 dark:text-gray-100">{analytics.analytics_chunks}</p>
            </div>
          </div>
        </div>

        {/* Sheets List */}
        {analytics.sheets && analytics.sheets.length > 0 && (
          <div className="bg-white dark:bg-zinc-800 rounded-xl shadow-lg p-6 border border-gray-200 dark:border-zinc-700">
            <div className="flex items-center mb-4">
              <Grid className="w-6 h-6 text-green-500 mr-3" />
              <h3 className="text-xl font-bold text-gray-800 dark:text-gray-100">Sheets</h3>
            </div>
            <div className="flex flex-wrap gap-2">
              {analytics.sheets.map((sheet, index) => (
                <div 
                  key={index}
                  className="px-4 py-2 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 rounded-lg border border-blue-200 dark:border-blue-700"
                >
                  {sheet}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Detailed Stats */}
        {analytics.detailed_stats && (
          <div className="bg-white dark:bg-zinc-800 rounded-xl shadow-lg p-6 border border-gray-200 dark:border-zinc-700">
            <div className="flex items-center mb-4">
              <Database className="w-6 h-6 text-purple-500 mr-3" />
              <h3 className="text-xl font-bold text-gray-800 dark:text-gray-100">Detailed Statistics</h3>
            </div>
            <div className="space-y-4">
              {Object.entries(analytics.detailed_stats).map(([sheet, stats]) => (
                <div key={sheet} className="border-l-4 border-purple-500 pl-4">
                  <h4 className="font-semibold text-gray-800 dark:text-gray-100 mb-2">{sheet}</h4>
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <p className="text-gray-500 dark:text-gray-400">Rows</p>
                      <p className="font-semibold text-gray-800 dark:text-gray-100">{stats.total_rows}</p>
                    </div>
                    <div>
                      <p className="text-gray-500 dark:text-gray-400">Columns</p>
                      <p className="font-semibold text-gray-800 dark:text-gray-100">{stats.total_columns}</p>
                    </div>
                    <div>
                      <p className="text-gray-500 dark:text-gray-400">Insights</p>
                      <p className="font-semibold text-gray-800 dark:text-gray-100">{stats.insights_count}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderAnalyticsSummaries = () => {
    if (!analytics || !analytics.analytics_summaries || analytics.analytics_summaries.length === 0) {
      return (
        <div className="bg-white dark:bg-zinc-800 rounded-xl shadow-lg p-8 border border-gray-200 dark:border-zinc-700 text-center">
          <Info className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-500 dark:text-gray-400">No analytics summaries available</p>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        {analytics.analytics_summaries.map((summary, index) => (
          <div 
            key={index}
            className="bg-white dark:bg-zinc-800 rounded-xl shadow-lg p-6 border border-gray-200 dark:border-zinc-700"
          >
            <div className="flex items-start mb-3">
              <Activity className="w-5 h-5 text-blue-500 mr-2 mt-1 flex-shrink-0" />
              <h4 className="text-lg font-semibold text-gray-800 dark:text-gray-100">Summary {index + 1}</h4>
            </div>
            <div className="bg-gray-50 dark:bg-zinc-900 rounded-lg p-4 border border-gray-200 dark:border-zinc-700">
              <pre className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap font-mono">
                {summary}
              </pre>
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderVisualizations = () => {
    if (!visualizations || !visualizations.charts || visualizations.charts.length === 0) {
      return (
        <div className="bg-white dark:bg-zinc-800 rounded-xl shadow-lg p-8 border border-gray-200 dark:border-zinc-700 text-center">
          <PieChart className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-500 dark:text-gray-400">No visualization suggestions available</p>
        </div>
      );
    }

    const chartIcons = {
      histogram: BarChart3,
      box_plot: Activity,
      bar_chart: BarChart3,
      pie_chart: PieChart,
      line_chart: TrendingUp,
      scatter_plot: Activity,
      heatmap: Grid
    };

    return (
      <div className="space-y-4">
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 border border-blue-200 dark:border-blue-700 mb-6">
          <p className="text-sm text-blue-800 dark:text-blue-200">
            <strong>Total Suggestions:</strong> {visualizations.total_suggestions}
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {visualizations.charts.map((chart, index) => {
            const Icon = chartIcons[chart.type] || BarChart3;
            return (
              <div 
                key={index}
                className="bg-white dark:bg-zinc-800 rounded-xl shadow-lg p-6 border border-gray-200 dark:border-zinc-700 hover:shadow-xl transition-shadow"
              >
                <div className="flex items-center mb-3">
                  <Icon className="w-6 h-6 text-purple-500 mr-3" />
                  <h4 className="text-lg font-semibold text-gray-800 dark:text-gray-100">{chart.title}</h4>
                </div>
                <div className="mb-2">
                  <span className="inline-block px-3 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 text-xs font-semibold rounded-full">
                    {chart.type.replace('_', ' ').toUpperCase()}
                  </span>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400">{chart.description}</p>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-zinc-900 dark:to-zinc-950 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-16 h-16 text-blue-500 animate-spin mx-auto mb-4" />
          <p className="text-gray-600 dark:text-gray-400 text-lg">Loading Excel Analytics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-zinc-900 dark:to-zinc-950 flex items-center justify-center">
        <div className="bg-white dark:bg-zinc-800 rounded-xl shadow-lg p-8 max-w-md border border-red-200 dark:border-red-700">
          <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-red-700 dark:text-red-300 mb-2 text-center">Error Loading Analytics</h2>
          <p className="text-red-600 dark:text-red-400 text-center mb-6">{error}</p>
          <button
            onClick={onBack}
            className="w-full bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg font-medium transition-colors"
          >
            Go Back
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-zinc-900 dark:to-zinc-950">
      {/* Header */}
      <div className="bg-white dark:bg-zinc-900 shadow-md border-b border-gray-200 dark:border-zinc-800">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <button
                onClick={onBack}
                className="mr-4 p-2 hover:bg-gray-100 dark:hover:bg-zinc-800 rounded-lg transition-colors"
              >
                <ArrowLeft className="w-6 h-6 text-gray-700 dark:text-gray-300" />
              </button>
              <div>
                <h1 className="text-3xl font-bold text-gray-800 dark:text-gray-100">Excel Analytics</h1>
                <p className="text-gray-600 dark:text-gray-400 mt-1">{documentName || 'Document Analysis'}</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <BarChart3 className="w-8 h-8 text-blue-500" />
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="bg-white dark:bg-zinc-900 border-b border-gray-200 dark:border-zinc-800">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex gap-1">
            <button
              onClick={() => setActiveTab('overview')}
              className={`px-6 py-3 font-semibold border-b-2 transition-colors ${
                activeTab === 'overview'
                  ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'
              }`}
            >
              Overview
            </button>
            <button
              onClick={() => setActiveTab('summaries')}
              className={`px-6 py-3 font-semibold border-b-2 transition-colors ${
                activeTab === 'summaries'
                  ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'
              }`}
            >
              Analytics Summaries
            </button>
            <button
              onClick={() => setActiveTab('visualizations')}
              className={`px-6 py-3 font-semibold border-b-2 transition-colors ${
                activeTab === 'visualizations'
                  ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'
              }`}
            >
              Visualizations
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        {activeTab === 'overview' && renderOverview()}
        {activeTab === 'summaries' && renderAnalyticsSummaries()}
        {activeTab === 'visualizations' && renderVisualizations()}
      </div>
    </div>
  );
};

export default ExcelAnalytics;