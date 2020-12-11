using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace Categorize_Support_Issues
{

    public class Issues   // This is our input dataset class and has 4 string fields 
    {
        [LoadColumn(0)]
        public string ID { get; set; }
        [LoadColumn(1)]
        public string Area { get; set; }
        [LoadColumn(2)]
        public string Title { get; set; }
        [LoadColumn(3)]
        public string Description { get; set; }
    }

    // This is the class we will use for prediction after the model has been trained.  
    // It has a single string(Area) and a "PredictLabel ColumnName" attribute.
    // PredictLabel will be used during prediction and evaluation. 
    public class IssuePrediction                                   
    {
        [ColumnName("PredictedLabel")]
        public string Area;
    }
}
