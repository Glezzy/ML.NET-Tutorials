using System;
using System.IO;
using System.Linq;
using Microsoft.ML;

namespace Categorize_Support_Issues
{
    class Program
    {
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);

        private static string _trainDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "issues_train.tsv");

        private static string _testDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "issues_test.tsv");

        private static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

        private static MLContext _mlContext;
        private static PredictionEngine<Issues, IssuePrediction> _predEngine;
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;

        // MLContext, when initialized, creates a new ML.NET environment that can be shared across the model creation workflow objects. Similar to DBContext in Entity Framework
        static void Main(string[] args)
        {
            // We will initialize mlcontext with a random seed of 0. Similar to doing Data science in Python so we can get repeatable results. 
            _mlContext = new MLContext(seed: 0);

            Console.WriteLine($"=============== Loading Dataset  ===============");
            // Initializing and loading _trainDataView as a global variable so it can be used in our pipeline.
            _trainingDataView = _mlContext.Data.LoadFromTextFile<Issues>(_trainDataPath, hasHeader: true);
            // Process data will extract and transform the data and then will return the processing pipeline.
            var pipeline = ProcessData();
        }

        public static IEstimator<ITransformer> ProcessData()
        {
            Console.WriteLine($"=============== Processing Data ===============");

            // Here we are transforming the Title and Description columns into a numeric vector for each called TitleFeaturized and DescriptionFeaturized.
            // Then we will append the featurization for both columns to the pipeline.
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")

                            .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                            .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                            // Lastly we shall combine all the feature columns into the Features column using Concatenate()
                            .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))


        }
    }
}
