#include<bits/stdc++.h>
using namespace std;
vector<string> split ( string s, char delim)
{
    vector<string> result;
    stringstream ss (s);
    string item;
    while (getline (ss, item, delim))
    {
        result.push_back (item);
    }
    return result;
}

int main()
{
    string line;
    ifstream fin;
    vector<string> dataset ;
    fin.open("dataset.txt");
    int i,count1=0,nf=0,nc=0,ndata=0;
    while (fin)
    {
        getline(fin, line);
        dataset.push_back(line);
    }

    fin.close();

    vector<string> result;
    result=split(dataset[0],' ');

    stringstream ss1(result[0]),ss2(result[1]),ss3(result[2]);
    ss1>> nf;
    ss2 >> nc;
    ss3 >> ndata;
    cout<<nf<<" "<<nc<<" "<<ndata<<endl;
    int l=0;
    vector<float> data[nc][nf];
    int ccount[nc];
    for(int i=0; i<nc; i++)
    {
        ccount[i]=0;
    }
    for(int i=1; i<dataset.size()-1; i++)
    {
        result=split(dataset[i],' ');
        vector<float> linedata;
        for(int j=0; j<=nf; j++)
        {
            stringstream ss(result[j]);
            float val=0;
            ss>>val;
            linedata.push_back(val);
        }
        int clas=(int)linedata[nf];
        ccount[clas]++;
        //cout<<clas<<endl;
        for(int k=0; k<nf; k++)
        {
            data[clas][k].push_back(linedata[k]);
        }
        l++;
    }

    ///prior
    float p[nc];
    for(int i=0; i<nc; i++)
    {
        p[i]=((float)ccount[i]/(float)l);
        cout<<p[i]<<endl;
    }

    float miu[nc][nf],miu_f[nf];
    float sum=0,avg=0;
    cout<<"class and feature wise mean:\n\n";
    for(int c=0; c<nc; c++)
    {
        for(int f=0; f<nf; f++)
        {
            for(int n=0; n<data[c][f].size(); n++)
            {
                sum=sum+data[c][f][n];
            }
            avg=sum/data[c][f].size();
            cout<<"class-"<<c<<" feature-"<<f<<" mean value: "<<avg<<endl;
            miu[c][f]=avg;
            sum=0;
            avg=0;
        }
        cout<<endl;
    }

    cout<<"\nfeature wise mean:\n";
    for(int ff=0; ff<nf; ff++)
    {
        for(int cc=0; cc<nc; cc++)
        {
            for(int nn=0; nn<data[cc][ff].size(); nn++)
            {
                sum=sum+data[cc][ff][nn];
            }
        }
        avg=sum/ndata;
        cout<<"feature-"<<ff<<" mean value: "<<avg<<endl;
        miu_f[ff]=avg;
        avg=0;
        sum=0;
    }

    float ssum=0,savg=0;
    float sigmac[nc][nf],sigmaf[nf];
    cout<<"\n\nclass and feature standard deviation:\n\n";
    for(int c=0; c<nc; c++)
    {
        for(int f=0; f<nf; f++)
        {
            for(int n=0; n<data[c][f].size(); n++)
            {
                ssum=ssum+pow(data[c][f][n]-miu[c][f],2);
            }
            savg=ssum/data[c][f].size();
            cout<<"class-"<<c<<" feature-"<<f<<" standard deviation value: "<<sqrt(savg)<<endl;
            sigmac[c][f]=sqrt(savg);
            ssum=0;
            savg=0;
        }
        cout<<endl;
    }
    cout<<"\n\nfeature wise standard deviation:\n\n";
    for(int ff=0; ff<nf; ff++)
    {
        for(int cc=0; cc<nc; cc++)
        {
            for(int nn=0; nn<data[cc][ff].size(); nn++)
            {
                ssum=ssum+pow(data[cc][ff][nn]-miu_f[ff],2);
            }
        }
        savg=ssum/ndata;
        cout<<"feature-"<<ff<<" standard deviation value: "<<sqrt(savg)<<endl;
        sigmaf[ff]=sqrt(savg);
        savg=0;
        ssum=0;
        cout<<endl;
    }



    fin.open("Test.txt");
    while (fin)
    {
        getline(fin, line);
        dataset.push_back(line);
    }
    fin.close();
    vector<float> linedata[dataset.size()];

    for(int i=0; i<dataset.size(); i++)
    {
        result=split(dataset[i],' ');
        for(int j=0; j<=nf; j++)
        {
            stringstream ss(result[j]);
            float val=0;
            ss>>val;
            linedata[i].push_back(val);
        }
        int clas=(int)linedata[i][nf];

        float p_fc[nf][nc];
        for(int k=0; k<nf; k++)
        {
            for(int l=0; l<nc; l++)
            {
                p_fc[k][l]=(exp(-pow((linedata[i][k]-miu[l][k]),2)/(2*sigmac[l][k]*sigmac[l][k])))/(sqrt(2*3.1416*sigmac[l][k]*sigmac[l][k]));
            }
        }

    }


    return 0;
}
