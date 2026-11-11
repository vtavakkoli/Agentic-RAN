/*
 * ns-3 EVENT-QUEUE DEMONSTRATOR for the JAGC JSON-line protocol.
 *
 * ACTION DEPENDENT: ns3::Simulator schedules each allocation, and the
 * next queue, reward and measured (MODELED) KPIs depend on that action.
 * This is NOT 5G-LENA, 3GPP NR MAC/PHY, or live Colosseum. Capacity is
 * declared by CLI rather than read from a configured NrGnbPhy.
 *
 * To obtain REAL NR results, replace UpdateQueues with PHY/MAC actuation
 * (e.g. 5G-LENA NrMacSchedulerOfdmaAi/OpenGym), report the observed BWP
 * capacity, and do not set backend=ns3-5glena until verified end-to-end.
 */
#include "ns3/core-module.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <regex>
#include <sstream>
#include <string>

using namespace ns3;
namespace {
constexpr std::array<const char*,3> kNames = {"eMBB","mMTC","URLLC"};
constexpr std::array<const char*,3> kSched = {
    "round_robin","waterfilling","proportional_fair"
};
std::uint32_t totalPrbs=52, reservedPrbs=4, maxSteps=20, t=0;
std::array<double,3> queue, arrival, channel, serviced;
std::array<std::uint32_t,3> allocated;
std::array<int,3> sched;
std::string scenario;
std::mt19937 rng;

void Error(const char* message) {
    std::cout << "{\"type\":\"error\",\"reason\":\"" << message << "\"}" << std::endl;
}
void Reset(std::uint32_t seed, const std::string& mode) {
    Simulator::Destroy(); rng.seed(seed); t=0; scenario=mode;
    std::uniform_real_distribution<double> variation(.84,1.16);
    const double load=mode=="congested"?1.6:1.;
    const double ch=mode=="weak_channel"?.55:1.;
    for (int i=0;i<3;++i) {
        queue[i]=(12000-3500*i)*variation(rng);
        arrival[i]=(10500-3200*i)*load;
        channel[i]=ch*variation(rng);
        allocated[i]=std::max(1u,(totalPrbs-reservedPrbs)/3);
        serviced[i]=0.;
        sched[i]=2;
    }
}
void UpdateQueues(std::array<std::uint32_t,3> allocation,
                  std::array<int,3> scheduling) {
    std::normal_distribution<double> fluctuation(0.,.025);
    for (int i=0;i<3;++i) {
        const double factor=scheduling[i]==2?1.:scheduling[i]==1?1.05:.92;
        serviced[i]=std::min(queue[i]+arrival[i],
                820.*allocation[i]*channel[i]*factor);
        const double burst=scenario=="bursty"&&t%4==3?1.65:1.;
        queue[i]=std::max(0.,queue[i]+burst*arrival[i]-serviced[i]);
        channel[i]=std::clamp(.94*channel[i]+.06+fluctuation(rng),.2,1.28);
        allocated[i]=allocation[i]; sched[i]=scheduling[i];
    }
    ++t;
}
void Observation(std::ostream& os) {
    os << std::fixed << std::setprecision(6);
    os << "{\"type\":\"observation\",\"step\":" << t << ",\"slices\":{";
    for (int i=0;i<3;++i) {
        if (i) os << ",";
        const double request=std::max(1.,arrival[i]/820.);
        const double granted=std::min(request,double(allocated[i]));
        const double cqi=std::clamp(channel[i]*9.,1.,15.);
        const double errors=std::clamp(18.-cqi,0.,100.);
        const double throughput=serviced[i]/100000.;
        os << "\"" << kNames[i] << "\":{\"metrics\":{"
           << "\"num_ues\":3,\"slice_prb\":" << allocated[i]
           << ",\"power_multiplier\":1,"
           << "\"scheduler_code\":" << sched[i]
           << ",\"dl_mcs\":" << cqi
           << ",\"dl_buffer_bytes\":" << queue[i]
           << ",\"dl_bitrate_mbps\":" << throughput
           << ",\"dl_errors_pct\":" << errors
           << ",\"dl_cqi\":" << cqi
           << ",\"ul_mcs\":" << cqi
           << ",\"ul_buffer_bytes\":" << queue[i]*.3
           << ",\"ul_bitrate_mbps\":" << throughput*.2
           << ",\"ul_errors_pct\":" << errors
           << ",\"ul_sinr\":" << channel[i]*10.
           << ",\"requested_prbs\":" << request
           << ",\"granted_prbs\":" << granted
           << ",\"grant_ratio\":" << granted/request << "}}";
    }
    os << "}}";
}
bool ParseAction(const std::string& line, std::array<std::uint32_t,3>& out,
                 std::array<int,3>& scheduling) {
    for (int i=0;i<3;++i) {
        const std::string pattern=std::string("\\\"")+kNames[i]+
            "\\\"\\s*:\\s*\\{\\s*\\\"prbs\\\"\\s*:\\s*([0-9]+)\\s*,"
            "\\s*\\\"scheduler\\\"\\s*:\\s*\\\"([a-z_]+)\\\"";
        std::smatch m;
        if (!std::regex_search(line,m,std::regex(pattern))) return false;
        try { out[i]=static_cast<std::uint32_t>(std::stoul(m[1])); }
        catch (...) { return false; }
        scheduling[i]=-1;
        for (int j=0;j<3;++j)
            if (m[2]==kSched[j]) scheduling[i]=j;
        if (scheduling[i]<0||out[i]<1) return false;
    }
    return std::uint64_t(out[0])+out[1]+out[2]<=totalPrbs-reservedPrbs;
}
bool ParseReset(const std::string& line, std::uint32_t& seed, std::string& mode) {
    std::smatch m;
    if (!std::regex_search(line,m,std::regex("\\\"seed\\\"\\s*:\\s*([0-9]+)"))) return false;
    try { seed=static_cast<std::uint32_t>(std::stoul(m[1])); }
    catch (...) { return false; }
    if (!std::regex_search(line,m,std::regex("\\\"scenario\\\"\\s*:\\s*\\\"([a-z_]+)\\\""))) return false;
    mode=m[1];
    return mode=="normal"||mode=="congested"||mode=="bursty"||mode=="weak_channel";
}
}
int main(int argc,char**argv) {
    CommandLine cmd(__FILE__);
    cmd.AddValue("totalPrbs","Declared demo capacity, NOT PHY measured",totalPrbs);
    cmd.AddValue("reservedPrbs","Declared demo reserved PRBs",reservedPrbs);
    cmd.AddValue("maxSteps","Maximum steps per reset",maxSteps);
    cmd.Parse(argc,argv);
    if (totalPrbs<3||totalPrbs>275||reservedPrbs>=totalPrbs||
        totalPrbs-reservedPrbs<3||maxSteps<1) return 2;
    std::cout << "{\"type\":\"hello\",\"protocol\":\"jagc-closed-loop-v1\","
      << "\"backend\":\"ns3-event-demo\",\"real_actuation\":false,\"cell\":{"
      << "\"carrier_id\":\"demo-carrier\",\"bwp_id\":\"demo-bwp\","
      << "\"total_prbs\":" << totalPrbs
      << ",\"reserved_prbs\":" << reservedPrbs
      << ",\"numerology\":1,\"source\":\"configured_ns3_demo\"}}" << std::endl;
    std::string line;
    while (std::getline(std::cin,line)) {
        if (line.find("\"type\":\"reset\"")!=std::string::npos) {
            std::uint32_t seed=0; std::string mode;
            if (!ParseReset(line,seed,mode)) {Error("invalid_reset");continue;}
            Reset(seed,mode); Observation(std::cout);std::cout<<std::endl;
        } else if (line.find("\"type\":\"step\"")!=std::string::npos) {
            std::array<std::uint32_t,3> alloc{};
            std::array<int,3> scheduling{};
            if (t>=maxSteps||!ParseAction(line,alloc,scheduling)) {
                Error("invalid_or_overbudget_action");continue;
            }
            Simulator::Schedule(MilliSeconds(100),&UpdateQueues,alloc,scheduling);
            Simulator::Stop(Simulator::Now()+MilliSeconds(100));
            Simulator::Run();
            const double cost=.30*queue[0]/20000.+.10*queue[1]/12000.+.60*queue[2]/8000.;
            std::cout << std::fixed << std::setprecision(6)
              << "{\"type\":\"transition\",\"reward\":" << 1.-cost
              << ",\"done\":" << (t>=maxSteps?"true":"false")
              << ",\"observation\":";
            Observation(std::cout);std::cout << "}" << std::endl;
        } else Error("invalid_command");
    }
    Simulator::Destroy();
}
