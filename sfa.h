#include "model.h"
#include "math.h"
#include "stats.h"

#include <string>
#include <fstream>
#include <vector>
#include <utility> // std::pair




class SFA : public Model
{
    double alpha = 0.001f;

    double gamma_long = 0.0f,gamma_short = 0.0f;
    double lambda_long = 0.0f,lambda_short = 0.0f;

    std::vector<double> x_tilde_vector;
    std::vector<double> x_bar_vector;
    std::vector<double> del_weight_vector;

    std::vector<double> del_weight1_vector;
    std::vector<double> del_weight2_vector;

    std::vector<double> input_vector;


    std::vector<double> y1_vector;
    std::vector<double> y2_vector;

    double y_tilde,y_bar,U,V;

    double y1_tilde, y2_tilde, y1_bar,y2_bar,U1,U2,V1,V2;
    double y1,y2;

    unsigned NUM_INPUT_NEURONS_X;
    unsigned NUM_INPUT_NEURONS_Y;
    unsigned TIMES_TO_RUN = 1;
    double ro = 450.0f; //period/phase
    int TOTAL_TIME = 8000*ro;
    vector<double> signal;
    vector<double> resultVector;
    vector<double> resultVector1;
    vector<double> resultVector2;

    vector<double> inputVector;
    vector<double> resultUni; // please change to last result
    vector<double> resultTuple;
    bool bAdapt = false;
    bool bForward = true;
    double y = 0.0f;


    std::vector<double> signalVector;
    std::vector<double> signalVector1;
    std::vector<double> signalVector2;

    public:
    SFA()
    {
        int NUM_INVARIANCES = 2;
        
        if(NUM_INVARIANCES == 1)
        {
            lambda_long = 2.0f * ro;
            lambda_short = ro/31.0f;
            gamma_long = pow(0.5f,(1.0f/(lambda_long)));
            gamma_short = pow(0.5f,(1.0f/(lambda_short)));

            y_tilde = 0.0f;
            y_bar = 0.0f;
            U = 0.000f;
            V = 0.000f;

            vector<unsigned> a;
            a.push_back(NUM_INPUT_NEURONS_X);
            a.push_back(1);
            SetTopology(a);
            InitializeTopology();

            for(int i=0;i<NUM_INPUT_NEURONS_X;i++)
            {
                x_bar_vector.push_back(0.0f);
                x_tilde_vector.push_back(0.0f);
                del_weight_vector.push_back(0.0f);
                input_vector.push_back(0.0f);
            }
        }
        else if(NUM_INVARIANCES == 2)
        {
            NUM_INPUT_NEURONS_Y = 51;
            NUM_INPUT_NEURONS_X = 51;

            lambda_long = 2.0f * ro;
            lambda_short = ro/31.0f;
            gamma_long = pow(0.5f,(1.0f/(lambda_long)));
            gamma_short = pow(0.5f,(1.0f/(lambda_short)));

            y1_tilde = 0.0f;
            y1_bar = 0.0f;
            y2_tilde = 0.0f;
            y2_bar = 0.0f;
            U1 = 0.000f;
            U2 = 0.0f;
            V1 = 0.0f;
            V2 = 0.0f;

            y1 = 0.0f;
            y2 = 0.0f;


            vector<unsigned> a;
            a.push_back(NUM_INPUT_NEURONS_X * NUM_INPUT_NEURONS_Y);
            a.push_back(2);
            SetTopology(a);
            InitializeTopology();

            for(int i=0;i<NUM_INPUT_NEURONS_X * NUM_INPUT_NEURONS_Y;i++)
            {
                x_bar_vector.push_back(0.0f);
                x_tilde_vector.push_back(0.0f);
                del_weight_vector.push_back(0.0f);
                del_weight1_vector.push_back(0.0f);
                del_weight2_vector.push_back(0.0f);
                input_vector.push_back(0.0f);
            }
        }


        cout<<"finished constructing"<<endl;
    }


    double GetCorrelationTrace(vector<double> va, vector<double> vb,int t)
    {
        int time_step = t;
        int lambda_correlation = fmin(100*ro,time_step);
        vector<double> vector_y1;
        vector<double> vector_y2;

        for(int i=resultVector1.size()-lambda_correlation;i<resultVector1.size();i++)
        {
            if(i>-1)
            {
                vector_y1.push_back(va[i]);
                vector_y2.push_back(vb[i]);
            }
        }
        return pearsoncoeff(vector_y1,vector_y2);
    }






    void write_csv(std::string filename, std::vector<std::pair<std::string, std::vector<double>>> dataset)
    {
    // Make a CSV file with one or more columns of integer values
    // Each column of data is represented by the pair <column name, column data>
    //   as std::pair<std::string, std::vector<int>>
    // The dataset is represented as a vector of these columns
    // Note that all columns should be the same size
    
    // Create an output filestream object
    std::ofstream myFile(filename);
    
    // Send column names to the stream
    for(int j = 0; j < dataset.size(); ++j)
    {
        myFile << dataset.at(j).first;
        if(j != dataset.size() - 1) myFile << ","; // No comma at end of line
    }
    myFile << "\n";
    
    // Send data to the stream
    for(int i = 0; i < dataset.at(0).second.size(); ++i)
    {
        for(int j = 0; j < dataset.size(); ++j)
        {
            myFile << dataset.at(j).second.at(i);
            if(j != dataset.size() - 1) myFile << ","; // No comma at end of line
        }
        myFile << "\n";
    }
    
    // Close the file
    myFile.close();
    }



    void Train()
    {
        alpha = 0.0f;
        for(int t=0;t<4.0f * ro;t++)
        {
            int v = GetSignalValue(t);
            signalVector.push_back(v);
            OscillateFeedForward(v,t);
        }
        alpha = 0.001f;
        for(int t = 4.0f * ro;t<(4.0f*ro+TOTAL_TIME);t++)
        {
            int v = GetSignalValue(t);
            signalVector.push_back(v);
            OscillateFeedForward(v,t);
            getNetwork()->UpdateWeights();
        }  
        
        std::vector<std::pair<std::string,
        std::vector<double>>> vals = {{"Values", resultVector}};
        write_csv("values.csv", vals);
        std::vector<std::pair<std::string,
        std::vector<double>>> vals2 = {{"j", getNetwork()->GetWeights()}};
        write_csv("weight_vector.csv", vals2);
    }


    void TrainTwoInvariances()
    {
        vector<double> corVector1,corVector2; //correlation vector. I should probably move this to the header.

        cout<<"started training"<<endl;
        alpha = 0.0f;
        for(int t=0;t<4.0f * ro;t++)
        {



            int* v = GetSignalTuple(t);
            signalVector1.push_back(v[0]);
            signalVector2.push_back(v[1]);


            OscillateFeedForwardTuple(v[0],v[1],t);




        }
        alpha = 0.001f;
        for(int t = 4.0f * ro;t<(4.0f*ro+TOTAL_TIME);t++)
        {
            int* v = GetSignalTuple(t);
            signalVector1.push_back(v[0]);
            signalVector2.push_back(v[1]);
            OscillateFeedForwardTuple(v[0],v[1],t);
            getNetwork()->UpdateWeights();



            corVector1.push_back(GetCorrelationTrace(signalVector1,resultVector2,t));
            corVector2.push_back(GetCorrelationTrace(signalVector2,resultVector2,t));
        }  
        
        std::vector<std::pair<std::string,
        std::vector<double>>> Output1 = {{"Values", resultVector1}};
        write_csv("output1.csv", Output1);

        std::vector<std::pair<std::string,
        std::vector<double>>> Output2 = {{"Values", resultVector2}};
        write_csv("output2.csv", Output2);
        
        std::vector<std::pair<std::string,
        std::vector<double>>> weights_two = {{"j", getNetwork()->GetWeights()}};
        write_csv("weight_vector.csv", weights_two);

        std::vector<std::pair<std::string,
        std::vector<double>>> O1 = {{"Values", signalVector1}};
        write_csv("signal1.csv", O1);

        std::vector<std::pair<std::string,
        std::vector<double>>> S2 = {{"Values", signalVector2}};
        write_csv("signal2.csv", S2);

        std::vector<std::pair<std::string,
        std::vector<double>>> C1 = {{"Values", corVector1}};
        write_csv("cor1.csv", C1);

        std::vector<std::pair<std::string,
        std::vector<double>>> C2 = {{"Values", corVector2}};
        write_csv("cor2.csv", C2);
    }

    int GetSignalValue(int time) //gets value of signal at time
    {
        int j = round(51.0f + 50.0f*sin(M_PI/180.0f*double(time)*360.0f / ro));
        input_vector.clear();
        for(int i=1;i<=NUM_INPUT_NEURONS_X;i++)
        {
            if(i != j)
                input_vector.push_back(0.0f);
            else if(i == j)
                input_vector.push_back(1.0f);
        }
        return j;
    }


    int* GetSignalTuple(int time) //gets value of signal at time
    {
        float phi = (float)M_PI/180.0f*time*17.0f/360.0f;
        int j1 = round(26.0f + 25.0f * sin((M_PI/180.0f*time*360.0f/ro) + phi));
        int j2 = round(26.0f + 25.0f * sin((M_PI/180.0f*time*360.0f/ro) - phi));

   //     cout<<j2<<endl;

        input_vector.clear();
        for(int i=1;i<=NUM_INPUT_NEURONS_Y;i++)
        {
            for(int j=1;j<=NUM_INPUT_NEURONS_X;j++)
            {
                if(j == j1 && i == j2)
                    input_vector.push_back(1.0f);
                else
                    input_vector.push_back(0.0f);
            }
        }
        int* tuple = new int[2];
        tuple[0] = j1;
        tuple[1] = j2;
        return tuple;
    }


    double CalculateDelWeights(int i)
    {
        double alphaV = alpha/V;
        double alphaU = alpha/U;

        if(isnan(alphaV))
            alphaV = 0.0f;
        if(isnan(alphaU))
            alphaU = 0.0f;
    
    
            
        double dely = y-y_bar;
        double delyt = y-y_tilde;
        double hebbian = alphaV * dely * (input_vector[i] - x_bar_vector[i]);
        double antihebbian = -1.0f*(alphaU * delyt * (input_vector[i] - x_tilde_vector[i]));
        del_weight_vector[i] = hebbian + antihebbian;

        return del_weight_vector[i];
    }

    double CalculateDelWeight(double v, double u, double output, double y_b, double y_ti, int i)
    {
        double alphaV = alpha/v;
        double alphaU = alpha/u;

        if(isnan(alphaV))
            alphaV = 0.0f;
        if(isnan(alphaU))
            alphaU = 0.0f;


               
        double dely = output-y_b;
        double delyt = output-y_ti;
        double hebbian = alphaV * dely * (input_vector[i] - x_bar_vector[i]);
        double antihebbian = -1.0f*(alphaU * delyt * (input_vector[i] - x_tilde_vector[i]));

        double del_weight = hebbian + antihebbian;
        return del_weight;
    }

    double getYBar(double yz, double y_barz)
    {
        return(gamma_long*y_barz + (1.0f-gamma_long)*yz); //last result
    }

    double getYTilde(double yz, double y_tildez)
    {
        double y_tilde;
        y_tilde = gamma_short*y_tildez + (1.0f-gamma_short)*yz;
        return y_tilde;
    }

    double getXBar(double x_barz,double x)
    {
        double x_bar;
        x_bar = gamma_long*x_barz + (1.0f-gamma_long)*x;
        return x_bar;
    }

    double getXTilde(double x_tildez, double x)
    {
        double t;
        t = gamma_short*x_tildez + (1.0f-gamma_short)*x;
        return t;
    }

    double getV(double Vz,double Y, double Y_BAR)
    {
        double t_V;
        t_V = gamma_long * Vz + (1.0f - gamma_long) * pow(Y - Y_BAR  ,2);
        return t_V;
    }

    double getU(double Uz, double Y, double Y_TILDE)
    {
        double t_U;
        t_U = gamma_long * Uz + (1.0f - gamma_long) * pow(Y - Y_TILDE,2);
        return t_U;
    }

    double GetOutput(int signal_value)
    {
        resultUni.clear();
        feedforward(GenerateInputs(signal_value)); //feed input vector into sfa model
        getNetwork()->getResults(resultUni);
        return resultUni.back();
    }

    void UpdateX(int i)
    {

        x_tilde_vector[i] = getXTilde(x_tilde_vector[i],input_vector[i]);
        x_bar_vector[i] = getXBar(x_bar_vector[i],input_vector[i]);
    }

    double GetY(int sig_val)
    {
        y = GetOutput(sig_val);
        resultVector.push_back(y);
        return y;
    }

    double* GetYTuple(int val1, int val2,int time_step)
    {
        double k = 10.0f;

        double* y_tuple;
        y_tuple = GetOutputTuple(val1,val2);



        y_tuple[1] = y_tuple[1] + k * GetWeightedAntiHebbian(time_step) * y_tuple[0];

        resultVector1.push_back(y_tuple[0]);
        resultVector2.push_back(y_tuple[1]);
        return y_tuple;

    }

    double GetWeightedAntiHebbian(int time_step)
    {
        int lambda_correlation = fmin(20*ro,time_step);
        vector<double> vector_y1;
        vector<double> vector_y2;

        for(int i=resultVector1.size()-lambda_correlation;i<resultVector1.size();i++)
        {
            if(i>-1)
            {
                vector_y1.push_back(resultVector1[i]);
                vector_y2.push_back(resultVector2[i]);
            }
        }

        

        double wah = -1.0f * pearsoncoeff(vector_y1,vector_y2);

        if(isnan(wah))
           wah = 0.0f;
        

        return wah;
    }

    double* GetOutputTuple(int val1,int val2)
    {

        vector<double> resultTuple;
        resultTuple.clear();
        feedforward(GenerateInputsFromTuple(val1,val2)); //feed input vector into sfa model
        
        //gets value at output nodes
        
        getNetwork()->getResults(resultTuple);

        double* rtuple = new double[2];
        rtuple[0] = resultTuple[0];
        rtuple[1] = resultTuple[1];

        return rtuple;
    }



    void UpdateNeuron(Neuron* neuron,int neuron_index)
    {
        neuron->m_outputWeights[0].setDW(del_weight_vector[neuron_index]);
    }

    void UpdateNeuronWithDelta(Neuron* neuron,double dw, int output_neuron_index)
    {
        neuron->m_outputWeights[output_neuron_index].setDW(dw);
    }

    void OscillateFeedForward(int signal_value, int time_step)
    {
        int sig_val = GetSignalValue(time_step);
        for(int neuron_index=0;neuron_index<NUM_INPUT_NEURONS_X;neuron_index++)
        {
            UpdateX(neuron_index);
        }
        V = getV(V,y,y_bar);
        U = getU(U,y,y_tilde);
        y = GetY(sig_val);
        y_bar = getYBar(y,y_bar);
        y_tilde = getYTilde(y,y_tilde);

        for(int neuron_index = 0;neuron_index<NUM_INPUT_NEURONS_X;neuron_index++)
        {
            Layer* inputLayer = &(getNetwork()->m_layers[0]);
            Neuron* thisNeuron = &(*inputLayer)[neuron_index];
            double dw = CalculateDelWeights(neuron_index);
            UpdateNeuron(thisNeuron,neuron_index);
        }
    }

    void OscillateFeedForwardTuple(int signal_value1,int signal_value2,int time_step)
    {

        for(int index=0;index<NUM_INPUT_NEURONS_X*NUM_INPUT_NEURONS_Y;index++)
        {
            UpdateX(index);
        }

        


        V1 = getV(V1,y1,y1_bar);
        V2 = getV(V2,y2,y2_bar);
        U1 = getU(U1,y1,y1_tilde);
        U2 = getU(U2,y2,y2_tilde);

        double* y_tuple = GetYTuple(signal_value1,signal_value2,time_step);

        y1 = y_tuple[0];
        y2 = y_tuple[1];

        y1_bar = getYBar(y_tuple[0],y1_bar);
        y1_tilde = getYTilde(y_tuple[0],y1_tilde);

        y2_bar = getYBar(y_tuple[1],y2_bar);
        y2_tilde = getYTilde(y_tuple[1],y2_tilde);



        for(int index1 = 0;index1<NUM_INPUT_NEURONS_Y * NUM_INPUT_NEURONS_X;index1++)
        {


                int neuron_index = index1;
                Layer* inputLayer = &(getNetwork()->m_layers[0]);
                Neuron* thisNeuron = &(*inputLayer)[neuron_index];
                double dw1 = CalculateDelWeight(V1,U1,y1,y1_bar,y1_tilde,neuron_index);
                double dw2 = CalculateDelWeight(V2,U2,y2,y2_bar,y2_tilde,neuron_index);
          
                UpdateNeuronWithDelta(thisNeuron,dw1,0);
                UpdateNeuronWithDelta(thisNeuron,dw2,1);

        }
    }      
    

    vector<double> GenerateInputs(int number_to_encode)                    
    {
        vector<double> inputVector; //one hot encoding
        inputVector.clear();
        for(int neuron_index=1;neuron_index<=NUM_INPUT_NEURONS_X;neuron_index++)
        {   
            if(number_to_encode == (neuron_index))
            {
                inputVector.push_back(1.0f);
            }
            else
            {
                inputVector.push_back(0.0f);
            }                
        }
        return inputVector;
    }

    vector<double> GenerateInputsFromTuple(int number1,int number2)
    {
        vector<double> inputVector; //one hot encoding
        inputVector.clear();
        for(int index1 = 1;index1 <= NUM_INPUT_NEURONS_Y; index1++)
        {   
            for(int index2 = 1;index2 <= NUM_INPUT_NEURONS_X; index2++)
            {
                if(number1 == index2 && number2 == index1)
                {
                    inputVector.push_back(1.0f);
                }
                else
                {
                    inputVector.push_back(0.0f);
                }                 
            }   
        }
        return inputVector;
    }
};