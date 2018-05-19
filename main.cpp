#include <iostream>
#include "src/cranium.h"
#include <iostream>
#include <cstdio>
#include <string>
#include <vector>

using namespace std;


struct WData
{
    float v[5] ;
    float y ;
};


void loaddatas( string filepath, vector<WData>& vec  )
{
    FILE* pf = fopen( filepath.c_str() , "r" ) ;
    char buff[256] ;
    int it = 0 ;
    while( fgets(buff , 256 , pf ) )
    {
        if( buff[0] == '#' ) continue ;
        if( strlen(buff) > 0 )
        {
            float v0,v1,v2,v3 ,v4,v5,v6;
            sscanf( buff , "%f %f %f %f %f %f %f" , &v0,&v1,&v2,&v3,&v4,&v5,&v6 ) ;
            //cout<<v0<< " "<<v1<<" "<<v2<<" "<<v3<<" "<<v4<<" "<<v5<<" "<<v6<<endl ;

            if( v0 < 0.0001 ) continue ;
            if( v1 < 0.0001 ) continue ;
            if( v2 < 0.0001 ) continue ;
            if( v3 < 0.0001 ) continue ;
            if( v4 < 0.0001 ) continue ;
            if( v5 < 0.0001 ) continue ;
            if( v6 < 0.0001 ) continue ;

            // if( v2 < 0.001 ) continue ; // for big net
            if( rand()%3 !=1 ) continue ;

            WData wd ;
            wd.v[0] = v0*100 ;
            wd.v[1] = v2*100  ;
            wd.v[2] = v3*100  ;
            wd.v[3] = v5*100  ;
            wd.v[4] = v6*100  ;

            wd.y = v1*100  ;
            vec.push_back(wd) ;


            ++ it ;
            if( it == 500 ) break ;

            if( it % 10000 ==0 )
            {
                cout<<"." ;
            }
        }
    }
    cout<<endl ;
    fclose(pf) ;
}



int main()
{
    cout << "green simu 3!" << endl;

    srand(time(NULL));


    string txtfiles[] = {
        "D:\\fy4-recent\\fy4truecolor\\modis-data-cross\\MOD021KM.A2018110.0120.061.2018110132438.hdf.datalist7.txt",
        "D:\\fy4-recent\\fy4truecolor\\modis-data-cross\\MOD021KM.A2018110.0125.061.2018110132406.hdf.datalist7.txt",
        "D:\\fy4-recent\\fy4truecolor\\modis-data-cross\\MOD021KM.A2018110.0245.061.2018110132400.hdf.datalist7.txt"
    } ;

    float** datas = 0 ;
    float** targets = 0 ;
    int nrows = 0 ;
    int ncols = 5 ;
    {
        vector<WData> wdVec ;
        for(int i = 0 ; i<3 ; ++ i )
        {
            loaddatas( txtfiles[i] , wdVec ) ;
            cout<<wdVec.size()<<" rows"<<endl ;
        }
        nrows = wdVec.size() ;
        cout<<"all data rows:"<<nrows<<endl ;

        datas = new float*[nrows] ;
        targets = new float*[nrows] ;
        for(int i = 0 ; i<nrows ; ++ i )
        {
            datas[i] = new float[ncols] ;
            targets[i] = new float[1] ;
            for(int j = 0 ; j<5 ; ++ j )
            {
                datas[i][j] = wdVec[i].v[j] ;
            }
            targets[i][0] = wdVec[i].y ;
        }
    }


    DataSet* trainingData =    createDataSet( nrows , ncols , datas   );
    DataSet* trainingClasses = createDataSet( nrows , 1, targets );

    size_t hiddenSize[] = {30};
    void (*hiddenActivations[])(Matrix*) = {relu};
    Network* network = createNetwork( 5, 1, hiddenSize, hiddenActivations, 1, linear);
    batchGradientDescent(network, trainingData, trainingClasses,
                         MEAN_SQUARED_ERROR,
                         20,
                         .01,
                         0,
                         0.001,
                         .9,
                         35000,
                         1,
                         1);


    cout<<"************* Test by input *************"<<endl;
    // load previous network from file
    //Network* previousNet = readNetwork("network");
    float** oneEx = (float**)malloc(sizeof(float*));
    oneEx[0] = (float*)malloc(sizeof(float)*5);
    DataSet* oneExData = createDataSet(1, 5, oneEx);

    float test0[] = { 6.11 , 8.35 , 26.55 , 23.7 , 9.72 } ;
    float test1[] = { 6.16 , 8.42 , 27.36 , 22.42 , 9.78 } ;
    float test2[] = { 6.46 , 10.17 ,29.90 , 26.29 , 13.20 } ;
    int it = 0 ;
    while( true )
    {
        float v[5] ;
        if( it == 0 )
        {
            for(int j = 0 ; j<5 ; ++j ) oneExData->data[0][j] = test0[j] ;
        }else if(it==1 )
        {
            for(int j = 0 ; j<5 ; ++j ) oneExData->data[0][j] = test1[j] ;
        }else if(it==2 )
        {
            for(int j = 0 ; j<5 ; ++j ) oneExData->data[0][j] = test2[j] ;
        }else{
            cout<<"input v0-5:"<<endl ;
            cin>>v[0]>>v[1]>>v[2]>>v[3]>>v[4] ;
            for(int j = 0 ; j<5 ; ++j )
            {
                oneExData->data[0][j] = v[j] ;
            }
        }
        forwardPassDataSet(network, oneExData);
        printf("simu:%6.4f\n",
               network->layers[network->numLayers - 1]->input->data[0]);
        ++ it ;
    }





    //destroyNetwork(previousNet);

    return 0;
}
